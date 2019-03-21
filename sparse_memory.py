
import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import numpy as np
import math

from util import *
import time


class SparseMemory(nn.Module):

  def __init__(
      self,
      input_size,
      mem_size=512,
      cell_size=32,
      independent_linears=False,
      read_heads=4,
      sparse_reads=4,
      num_lists=None,
      index_checks=None,
      gpu_id=-1,
      mem_gpu_id=-1,
      direct_write=False
      #x added direct write, independent false
  ):
    super(SparseMemory, self).__init__()

    self.mem_size = mem_size
    self.cell_size = cell_size
    self.gpu_id = gpu_id
    self.mem_gpu_id = mem_gpu_id
    self.input_size = input_size
    self.independent_linears = independent_linears
    self.K = sparse_reads if self.mem_size > sparse_reads else self.mem_size
    # if self.print_tensors: print(f"k: {self.K}")
    # if self.print_tensors: print(f"mem_size: {self.mem_size}")
    self.read_heads = read_heads
    self.num_lists = num_lists if num_lists is not None else int(self.mem_size / 100)+1
    self.index_checks = min(self.num_lists // 20, self.num_lists) if index_checks is None else index_checks
    self.direct_write = direct_write
    #n needs to be exchanged to true token lenght
    self.s = 2
    self.input_size = self.input_size // 2

    self.print_tensors = False
    self.usage_type = "lru"

    m = self.mem_size
    w = self.cell_size
    r = self.read_heads
    # The visible memory size: (K * R read heads, and least used memory cell)
    self.c = (self.K * r) + 1

    if self.independent_linears:
      if self.gpu_id != -1:
        self.read_query_transform = nn.Linear(self.input_size, w * r).cuda()
        self.write_vector_transform = nn.Linear(self.input_size, w).cuda()
        self.interpolation_gate_transform = nn.Linear(self.input_size, self.c).cuda()
        self.write_gate_transform = nn.Linear(self.input_size, 1).cuda()
      else:
        self.read_query_transform = nn.Linear(self.input_size, w * r)
        self.write_vector_transform = nn.Linear(self.input_size, w)
        self.interpolation_gate_transform = nn.Linear(self.input_size, self.c)
        self.write_gate_transform = nn.Linear(self.input_size, 1)
      T.nn.init.orthogonal(self.read_query_transform.weight)
      T.nn.init.orthogonal(self.write_vector_transform.weight)
      T.nn.init.orthogonal(self.interpolation_gate_transform.weight)
      T.nn.init.orthogonal(self.write_gate_transform.weight)
    else:
      #x depending on directly writing or not, create enough outputs
      if self.direct_write:
        #x read query w, sparse read plus lru gates, write gate
        self.interface_size = (w * r) + 1+ 1
      else:
        #x with addition write vector calculation
        self.interface_size = (w * r) + w + 1 + 1
      
      if self.gpu_id != -1:
        self.interface_weights = nn.Linear(self.input_size, self.interface_size).cuda()
      else:
        self.interface_weights = nn.Linear(self.input_size, self.interface_size)
      T.nn.init.orthogonal_(self.interface_weights.weight)

    # creates and 5x5 identitiy
    self.I = cuda(1 - T.eye(self.c).unsqueeze(0), gpu_id=self.gpu_id)  # (1 * n * n)
    self.δ = 0.005  # minimum usage
    self.timestep = 0
    self.mem_limit_reached = False
    if self.gpu_id != -1:
      self.cuda()

  def rebuild_indexes(self, hidden, erase=False):
    b = hidden["memory"].size(0)

    # if indexes already exist, we reset them
    if "indexes" in hidden:
      [x.reset() for x in hidden["indexes"]]
    else:

      try:
        # create new indexes, try to use FAISS, fall back to FLAN
        from faiss_index import FAISSIndex
        hidden["indexes"] = \
            [FAISSIndex(cell_size=self.cell_size,
                        nr_cells=self.mem_size, K=self.K, num_lists=self.num_lists,
                        probes=self.index_checks, gpu_id=self.mem_gpu_id) for x in range(b)]
      except Exception as e:
        print("\nFalling back to FLANN (CPU). \nFor using faster, GPU based indexes, install FAISS: `conda install faiss-gpu -c pytorch`")
        from flann_index import FLANNIndex
        hidden["indexes"] = \
            [FLANNIndex(cell_size=self.cell_size,
                        nr_cells=self.mem_size, K=self.K, num_kdtrees=self.num_lists,
                        probes=self.index_checks, gpu_id=self.mem_gpu_id) for x in range(b)]

    # add existing memory into indexes
    pos = hidden["read_positions"].squeeze().data.cpu().numpy()
    if not erase:
      for n, i in enumerate(hidden["indexes"]):
        i.reset()
        i.add(hidden["memory"][n], last=pos[n][-1])
        # else:
        #   i.reset()
        #   i.add(hidden["memory"][n], last=pos[-1])

    else:
      self.timestep = 0
      self.mem_limit_reached = False

    return hidden

  def reset(self, batch_size=1, hidden=None, erase=True):
    m = self.mem_size
    w = self.cell_size
    b = batch_size
    r = self.read_heads
    c = self.c
    s = self.s


    if hidden is None:
      hidden = {
          # warning can be a huge chunk of contiguous memory
          "memory": cuda(T.zeros(b, m, w).fill_(δ), gpu_id=self.mem_gpu_id).contiguous(),
          
          "visible_memory": cuda(T.zeros(b, c, w).fill_(δ), gpu_id=self.mem_gpu_id).contiguous(),

          "write_weights": cuda(T.zeros(b, s).fill_(δ), gpu_id=self.gpu_id),
          "read_vectors": cuda(T.zeros(b, r, w).fill_(δ), gpu_id=self.gpu_id),
          #n need to add one place for each readhead instead of just 1
          "least_used_mem": cuda(T.arange((c*s)+1, (c*s)+s+1).expand(b, s), gpu_id=self.gpu_id).long(),
          "usage": cuda(T.arange(0.001, 0, -(0.001/m)).expand(b, m), gpu_id=self.gpu_id).contiguous(),
          "read_positions": cuda(T.arange(0, c*s).expand(b, c*s), gpu_id=self.gpu_id).long(),
          #x lets each position head read a different position
          "read_pos_list": [ cuda(T.arange(x*c, (x+1)*c).expand(b, c), gpu_id=self.gpu_id).long() for x in range(s)]
          #n read gate should be added here
      }
      hidden = self.rebuild_indexes(hidden, erase=False)
    else:
      hidden["memory"] = hidden["memory"].clone()
      hidden["visible_memory"] = hidden["visible_memory"].clone()
      hidden["read_weights"] = hidden["read_weights"].clone()
      hidden["write_weights"] = hidden["write_weights"].clone()
      hidden["read_vectors"] = hidden["read_vectors"].clone()
      hidden["least_used_mem"] = hidden["least_used_mem"].clone()
      hidden["usage"] = hidden["usage"].clone()
      hidden["read_positions"] = hidden["read_positions"].clone()
      hidden = self.rebuild_indexes(hidden, erase)

      if erase:
        hidden["memory"].data.fill_(δ)
        hidden["visible_memory"].data.fill_(δ)
        hidden["read_weights"].data.fill_(δ)
        hidden["write_weights"].data.fill_(δ)
        hidden["read_vectors"].data.fill_(δ)
        hidden["least_used_mem"].data.fill_(c + 1)
        hidden["usage"].data.fill_(0)
        hidden["read_positions"] = cuda(
            T.arange(0, c).expand(b, c), gpu_id=self.gpu_id).long()
        hidden = self.rebuild_indexes(hidden, erase=False)
        self.timestep = 0

    return hidden

  def write_into_sparse_memory(self, hidden):
    visible_memory = hidden["visible_memory"]
    positions = hidden["read_positions"]

    # update memory
    hidden["memory"].scatter_(1, positions.unsqueeze(2).expand(self.b, self.vis_size, self.cell_size), visible_memory)

    # non-differentiable operations
    pos = positions.data.cpu().numpy()
    if self.print_tensors: print("pos start")
    if self.print_tensors: print(pos)
    #for p in pos: if self.print_tensors: print(p)
    if self.print_tensors: print("pos end")
    for batch in range(self.b):
      # update indexes
      if self.print_tensors: print("pos batch")
      if self.print_tensors: print(pos[batch][-1])
      hidden["indexes"][batch].reset()
      #n this could be changed to the old version
      # hidden["indexes"][batch].add(hidden["memory"][batch], last=(pos[batch][-1] if not self.mem_limit_reached else None))
      hidden["indexes"][batch].add(hidden["memory"][batch], last=None)
      # else:
      #        # update indexes

      #   hidden["indexes"][batch].reset()
      #   if self.print_tensors: print(f"read positions at sparse time: ")
      #   if self.print_tensors: print(hidden["read_positions"])
      #   if self.print_tensors: print("current pos")
      #   if self.print_tensors: print(pos[0][-1])
      #   if self.print_tensors: print(f"mem slots: {m}")
      #   hidden["indexes"][batch].add(hidden["memory"][batch], last=(pos[0][-1] if not self.mem_limit_reached else None))

    mem_limit_reached = hidden["least_used_mem"][0].data.cpu().numpy()[0] >= self.mem_size - 1
    self.mem_limit_reached = mem_limit_reached or self.mem_limit_reached

    return hidden

  def write(self, interpolation_gate, write_vector, write_gate, hidden):
    # take only the read weights out that were actually read on the previous f pass
    # (b * m) -> (b * c)
    read_weights = hidden["read_weights"]
    # encourage read and write in the first timestep
    if self.timestep == 1: read_weights =  read_weights + 1
    #if self.timestep == 2: read_weights =  read_weights + 1

    I, relevant_usages, usage = self.update_usage_before(
        hidden["read_positions"],
        hidden["usage"]
    )

    # either we write to previous read locations

    # # keep the interpolation gate fixed to the same order
    # _ , rw_sorted_indexes = T.sort(read_weights, descending=False)
    # rw_sorted_indexes = rw_sorted_indexes[:,:,:self.c].clone()
    # interpolation_gate = interpolation_gate.unsqueeze(3)
    # zeros = cuda(T.zeros(self.b, self.s, self.vis_size - self.c), gpu_id=self.gpu_id)

    # # interpolation_gate = T.cat((interpolation_gate, zeros), 2)

    # abc = interpolation_gate.gather(2, rw_sorted_indexes)
    interpolation_gate = interpolation_gate
    x = interpolation_gate * read_weights
    # or to a new location
    y = (1 - interpolation_gate) * I
    write_weights = write_gate * (x + y)

    # store the write weights
    # hidden["write_weights"].scatter_(1, hidden["read_positions"], write_weights)

    hidden["usage"] = self.update_usage_after(hidden["read_positions"], read_weights, write_weights, usage, relevant_usages)
    # erase matrix
    # combine erase matrixes for all heads
    I = T.sum(I, dim=1)
    I = T.ge(I,  T.ones(I.size())).float()
    erase_matrix = I.unsqueeze(2).expand(self.b, self.vis_size, self.cell_size)


    writings = T.matmul(write_weights.unsqueeze(3), write_vector)

    writings = T.sum(writings, dim=1)
    # write into memory
    hidden["visible_memory"] = hidden["visible_memory"] * \
        (1 - erase_matrix) + writings


    hidden = self.write_into_sparse_memory(hidden)
    # torch.set_printoptions(threshold=5000)
    # print(hidden["memory"][0].sum(1) > 0.05)
    # import pdb; pdb.set_trace()

    # update least used memory cell
    if self.print_tensors: print("usage before lum")
    if self.print_tensors: print(hidden["usage"])
    hidden["least_used_mem"] = T.topk(hidden["usage"], self.s, dim=-1, largest=False)[1]
    if self.print_tensors: print("leas used mem")
    if self.print_tensors: print(hidden["least_used_mem"])

    return hidden

  def update_usage_before(self, read_positions, usage):
    (b, n) = read_positions.size()

    # usage is timesteps since a non-negligible memory access


    # usage before write
    relevant_usages = usage.gather(1, read_positions)
    # indicator of words with minimal memory usage
    # takes the lowest usage of each batch and returns its values
    minusage = T.topk(relevant_usages, self.s, -1, largest=False)[0]
    minusage = minusage.view(b, self.s, 1)
    #minusage = T.min(relevant_usages, -1, keepdim=True)[0]
    minusage = minusage.expand(b, self.s, n)
    compareusage = relevant_usages.unsqueeze(1)
    # returns matrix with the minimum usage position as 1 all other positions as 0
    I = (compareusage == minusage).float()

    return I, relevant_usages, usage


  def update_usage_after(self, read_positions, read_weights, write_weights, usage, relevant_usages):
    (b, _) = read_positions.size()
    # usage is timesteps since a non-negligible memory access
    # read_weights_col = T.prod(read_weights, 1)
    # write_weights_col = T.prod(write_weights, 1)
    read_weights_col = T.sum(read_weights, 1)
    write_weights_col = T.sum(write_weights, 1)


    u = (T.abs(read_weights_col) + T.abs(write_weights_col) > self.δ).float()

    # usage before write

    # usage after write
    # maybe instead decaying relevant usages by using usages = usages * 0.5 + self.timestep
    # or just ture lru by doing usages = self.timestep for all acessed locations
    if self.usage_type == "original":
      relevant_usages = (self.timestep - relevant_usages) * u + relevant_usages * (1 - u)
    elif self.usage_type == "lru":
      relevant_usages = self.timestep * u + relevant_usages * (1 - u)
    else:
      print("Usage Type is necessary")
    
    #read_positions = T.tensor([[ 0,  3,  4,  9, 10], [0,0,0,0,0],[0,0,0,0,0 ],[ 0,0,0,0,0]])
    #usage = usage.clone().contiguous()
    # bug in pytorch when not calling contigous on scattered tensor

    usage.scatter_(dim = 1, index= read_positions, src= relevant_usages)


    return usage

  def read_from_sparse_memory(self, memory, indexes, keys, least_used_mem, usage):
    b = keys.size(0)
    s = keys.size(1)

    read_positions = []
    keys = keys.view(b, s* self.read_heads, -1)
    # we search for k cells per read head
    if self.print_tensors: print("sparse read now")
    for batch in range(b):
      distances, positions = indexes[batch].search(keys[batch])
      
      if self.print_tensors: print("bathc positions")
      if self.print_tensors: print(positions)
      if self.print_tensors: print(positions.size())
      read_positions.append(positions)
    if self.print_tensors: print(f"b: {b}")

    read_positions = T.stack(read_positions, 0)

    

    if self.print_tensors: print("read positions")
    if self.print_tensors: print(read_positions.size())
    if self.print_tensors: print(read_positions)

    # add least used mem to read positions
    # TODO: explore possibility of reading co-locations or ranges and such
    (b, r, k) = read_positions.size()
    #n this is the thing that combines all the reads heads that we dont want
    read_positions = var(read_positions)

    # no gradient here
    # temporal reads
    (b, m, w) = memory.size()
    # get the top KL entries
    #max_length = int(least_used_mem[0, 0].data.cpu().numpy()) if not self.mem_limit_reached else (m-1)
    max_length = m-1
    if self.print_tensors: print(f"max length: {max_length}")
    
    least_used_mem = least_used_mem.view(b,s,1)
    # differentiable ops
    # append forward and backward read positions, might lead to duplicates
    if self.print_tensors: print("read positions b")
    if self.print_tensors: print(read_positions)
    read_positions = T.cat([read_positions, least_used_mem], 2)
    if self.print_tensors: print("read positions c")
    if self.print_tensors: print(read_positions)
    # issue with batchsize 1
    #read_positions = T.clamp(read_positions, 0, max_length)
    if self.print_tensors: print("read positions d")
    if self.print_tensors: print(read_positions)
    read_positions = read_positions.view(b, -1)
    # deduplicate all read positions

    read_positions = torch.unique(read_positions, sorted=False, dim=1)
    self.vis_size = read_positions.size(1)
    
    # expand to get all the w dimension locations

    visible_memory = memory.gather(1, read_positions.unsqueeze(2).expand(b, self.vis_size, w))
    
    # take the vectors of the sparse reads and lru and let the read heads each look for the most similiar vector, then do softmax among all the vectors
    # for each head (b x r x (r*k + lru))
    # output shape (b x r x m), where m = r * K + 1
    read_weights = σ(θ(visible_memory, keys), 2)
    # let each head return one vector based on the previous softmax (b x r x w)
    read_vectors = T.bmm(read_weights, visible_memory)
    # collapses all heads into one average
    # (b x r x m) -> (b x m), where each element of m is the value of all read heads multiplied. This represents the average reading of an position

    #read_weights = T.prod(read_weights, 1)


    return read_vectors, read_positions, read_weights, visible_memory

  def read(self, read_query, hidden):
    # sparse read
    read_vectors, positions, read_weights, visible_memory = \
        self.read_from_sparse_memory(
            hidden["memory"],
            hidden["indexes"],
            read_query,
            hidden["least_used_mem"],
            hidden["usage"]
        )

    hidden["read_positions"] = positions
    #  use position = [2, 8 ,10] to put these sparse read location with their real read weights = [0,0,0.34,0,0,0,0,0,0,0.55,0,0.99]
    # updates the read weights only sparsely
    hidden["read_weights"] = read_weights
    # what we actually output
    hidden["read_vectors"] = read_vectors
    hidden["visible_memory"] = visible_memory

    return hidden["read_vectors"], hidden

  def forward(self, ξ, hidden):
    t = time.time()

    #x added fake double input
    #n need to remove again
    # ξ = torch.stack([ξ, ξ], dim=1)
    # ξ = ξ.detach()
    m = self.mem_size
    w = self.cell_size
    r = self.read_heads
    c = self.c
    b = ξ.size()[0]
    ξ = ξ.view(b, self.s, -1)
    self.b = b
    # s is the number of sequence tokens of input
    s = ξ.size()[1]
    self.s = s


    # if self.independent_linears:
    #   # r read keys (b * r * w)
    #   read_query = self.read_query_transform(ξ).view(b, r, w)
    #   # write key (b * 1 * w)
    #   write_vector = self.write_vector_transform(ξ).view(b, 1, w)
    #   # write vector (b * 1 * r)
    #   interpolation_gate = F.sigmoid(self.interpolation_gate_transform(ξ)).view(b, c)
    #   # write gate (b * 1)
    #   write_gate = F.sigmoid(self.write_gate_transform(ξ).view(b, 1))
    # else:
    ξ = self.interface_weights(ξ)
    # r read keys (b * r * w)
    read_query = ξ[:, :, :r * w].contiguous().view(b, s, r, w)
    # write key (b * 1 * w)
    if self.direct_write:
      write_vector = ξ
      # write vector (b * 1 * r)
      interpolation_gate = F.sigmoid(ξ[: , :, r * w: r * w + 1]).contiguous().view(b, s, 1)
      # write gate (b * 1)
      #n maybe need to change unsqueeze dim (changed it already from 1 to 2, but dont know)
      write_gate = F.sigmoid(ξ[: ,:, -1].contiguous()).view(b, s, 1)
    else:
      write_vector = ξ[:, :,  r * w: r * w + w].contiguous().view(b,s , 1, w)
      # write vector (b * 1 * r)
      interpolation_gate = F.sigmoid(ξ[:,:, r * w + w: r * w + w + 1]).contiguous().view(b, s, 1)
      # write gate (b * 1)
      
      write_gate = F.sigmoid(ξ[:, :, -1].contiguous()).view(b, s, 1)


    self.timestep += 1
    #x changed order to first read then write
    read_vectors, hidden = self.read(read_query, hidden)
    hidden = self.write(interpolation_gate, write_vector, write_gate, hidden)
    return read_vectors, hidden
    # hidden = self.write(interpolation_gate, write_vector, write_gate, hidden)
    # return self.read(read_query, hidden)
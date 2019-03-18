
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
      index_checks=32,
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
    self.num_lists = num_lists if num_lists is not None else int(self.mem_size / 100)
    self.index_checks = index_checks
    self.direct_write = direct_write

    self.print_tensors = False

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
        self.interface_size = (w * r) + self.c + 1
      else:
        #x with addition write vector calculation
        self.interface_size = (w * r) + w + self.c + 1
      
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
    b = hidden['memory'].size(0)

    # if indexes already exist, we reset them
    if 'indexes' in hidden:
      [x.reset() for x in hidden['indexes']]
    else:
      # create new indexes, try to use FAISS, fall back to FLAN
      from faiss_index import FAISSIndex
      hidden['indexes'] = \
          [FAISSIndex(cell_size=self.cell_size,
                      nr_cells=self.mem_size, K=self.K, num_lists=self.num_lists,
                      probes=self.index_checks, gpu_id=self.mem_gpu_id) for x in range(b)]
      # except Exception as e:
      #   print("\nFalling back to FLANN (CPU). \nFor using faster, GPU based indexes, install FAISS: `conda install faiss-gpu -c pytorch`")
      #   from flann_index import FLANNIndex
      #   hidden['indexes'] = \
      #       [FLANNIndex(cell_size=self.cell_size,
      #                   nr_cells=self.mem_size, K=self.K, num_kdtrees=self.num_lists,
      #                   probes=self.index_checks, gpu_id=self.mem_gpu_id) for x in range(b)]

    # add existing memory into indexes
    pos = hidden['read_positions'].squeeze().data.cpu().numpy()
    if not erase:
      for n, i in enumerate(hidden['indexes']):
        i.reset()
        i.add(hidden['memory'][n], last=pos[n][-1])
        # else:
        #   i.reset()
        #   i.add(hidden['memory'][n], last=pos[-1])

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

    if hidden is None:
      hidden = {
          # warning can be a huge chunk of contiguous memory
          'memory': cuda(T.zeros(b, m, w).fill_(δ), gpu_id=self.mem_gpu_id),
          
          'visible_memory': cuda(T.zeros(b, c, w).fill_(δ), gpu_id=self.mem_gpu_id),
          'read_weights': cuda(T.zeros(b, m).fill_(δ), gpu_id=self.gpu_id),
          'write_weights': cuda(T.zeros(b, m).fill_(δ), gpu_id=self.gpu_id),
          'read_vectors': cuda(T.zeros(b, r, w).fill_(δ), gpu_id=self.gpu_id),
          
          'least_used_mem': cuda(T.zeros(b, 1).fill_(c + 1), gpu_id=self.gpu_id).long(),
          'usage': cuda(T.zeros(b, m), gpu_id=self.gpu_id),
          'read_positions': cuda(T.arange(0, c).expand(b, c), gpu_id=self.gpu_id).long()
          #n read gate should be added here
      }
      hidden = self.rebuild_indexes(hidden, erase=False)
    else:
      hidden['memory'] = hidden['memory'].clone()
      hidden['visible_memory'] = hidden['visible_memory'].clone()
      hidden['read_weights'] = hidden['read_weights'].clone()
      hidden['write_weights'] = hidden['write_weights'].clone()
      hidden['read_vectors'] = hidden['read_vectors'].clone()
      hidden['least_used_mem'] = hidden['least_used_mem'].clone()
      hidden['usage'] = hidden['usage'].clone()
      hidden['read_positions'] = hidden['read_positions'].clone()
      hidden = self.rebuild_indexes(hidden, erase)

      if erase:
        hidden['memory'].data.fill_(δ)
        hidden['visible_memory'].data.fill_(δ)
        hidden['read_weights'].data.fill_(δ)
        hidden['write_weights'].data.fill_(δ)
        hidden['read_vectors'].data.fill_(δ)
        hidden['least_used_mem'].data.fill_(c + 1)
        hidden['usage'].data.fill_(0)
        hidden['read_positions'] = cuda(
            T.arange(0, c).expand(b, c), gpu_id=self.gpu_id).long()
        hidden = self.rebuild_indexes(hidden, erase=False)

    return hidden

  def write_into_sparse_memory(self, hidden):
    visible_memory = hidden['visible_memory']
    positions = hidden['read_positions']

    (b, m, w) = hidden['memory'].size()
    # update memory
    hidden['memory'].scatter_(1, positions.unsqueeze(2).expand(b, self.c, w), visible_memory)

    # non-differentiable operations
    pos = positions.data.cpu().numpy()
    if self.print_tensors: print("pos start")
    if self.print_tensors: print(pos)
    #for p in pos: if self.print_tensors: print(p)
    if self.print_tensors: print("pos end")
    for batch in range(b):
      # update indexes
      if self.print_tensors: print("pos batch")
      if self.print_tensors: print(pos[batch][-1])
      hidden['indexes'][batch].reset()
      # hidden['indexes'][batch].add(hidden['memory'][batch], last=(pos[batch][-1] if not self.mem_limit_reached else None))
      hidden['indexes'][batch].add(hidden['memory'][batch], last=None)
      # else:
      #        # update indexes

      #   hidden['indexes'][batch].reset()
      #   if self.print_tensors: print(f"read positions at sparse time: ")
      #   if self.print_tensors: print(hidden["read_positions"])
      #   if self.print_tensors: print("current pos")
      #   if self.print_tensors: print(pos[0][-1])
      #   if self.print_tensors: print(f"mem slots: {m}")
      #   hidden['indexes'][batch].add(hidden['memory'][batch], last=(pos[0][-1] if not self.mem_limit_reached else None))

    mem_limit_reached = hidden['least_used_mem'][0].data.cpu().numpy()[0] >= self.mem_size - 1
    self.mem_limit_reached = mem_limit_reached or self.mem_limit_reached

    return hidden

  def write(self, interpolation_gate, write_vector, write_gate, hidden):
    # take only the read weights out that were actually read on the previous f pass
    # (b * m) -> (b * c)
    read_weights = hidden['read_weights'].gather(1, hidden['read_positions'])
    # encourage read and write in the first timestep
    if self.timestep == 1: read_weights =  read_weights + 1
    #if self.timestep == 2: read_weights =  read_weights + 1
    write_weights = hidden['write_weights'].gather(1, hidden['read_positions'])

    hidden['usage'], I = self.update_usage(
        hidden['read_positions'],
        read_weights,
        write_weights,
        hidden['usage']
    )

    # either we write to previous read locations
    x = interpolation_gate * read_weights
    # or to a new location
    y = (1 - interpolation_gate) * I
    write_weights = write_gate * (x + y)

    # store the write weights
    hidden['write_weights'].scatter_(1, hidden['read_positions'], write_weights)

    # erase matrix
    erase_matrix = I.unsqueeze(2).expand(hidden['visible_memory'].size())

    # write into memory
    hidden['visible_memory'] = hidden['visible_memory'] * \
        (1 - erase_matrix) + T.bmm(write_weights.unsqueeze(2), write_vector)
    hidden = self.write_into_sparse_memory(hidden)

    # update least used memory cell
    if self.print_tensors: print("usage before lum")
    if self.print_tensors: print(hidden["usage"])
    hidden['least_used_mem'] = T.topk(hidden['usage'], 1, dim=-1, largest=False)[1]
    if self.print_tensors: print("leas used mem")
    if self.print_tensors: print(hidden['least_used_mem'])

    return hidden

  def update_usage(self, read_positions, read_weights, write_weights, usage):
    (b, _) = read_positions.size()
    # usage is timesteps since a non-negligible memory access
    u = (read_weights + write_weights > self.δ).float()

    # usage before write
    relevant_usages = usage.gather(1, read_positions)

    # indicator of words with minimal memory usage
    minusage = T.min(relevant_usages, -1, keepdim=True)[0]
    minusage = minusage.expand(relevant_usages.size())
    I = (relevant_usages == minusage).float()

    # usage after write
    relevant_usages = (self.timestep - relevant_usages) * u + relevant_usages * (1 - u)

    usage.scatter_(1, read_positions, relevant_usages)

    return usage, I

  def read_from_sparse_memory(self, memory, indexes, keys, least_used_mem, usage):
    b = keys.size(0)
    read_positions = []

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
    read_positions = var(read_positions).squeeze(1).view(b, -1)

    # no gradient here
    # temporal reads
    (b, m, w) = memory.size()
    # get the top KL entries
    #max_length = int(least_used_mem[0, 0].data.cpu().numpy()) if not self.mem_limit_reached else (m-1)
    max_length = m-1
    if self.print_tensors: print(f"max length: {max_length}")

    # differentiable ops
    # append forward and backward read positions, might lead to duplicates
    if self.print_tensors: print("read positions b")
    if self.print_tensors: print(read_positions)
    read_positions = T.cat([read_positions, least_used_mem], 1)
    if self.print_tensors: print("read positions c")
    if self.print_tensors: print(read_positions)
    # issue with batchsize 1
    read_positions = T.clamp(read_positions, 0, max_length)
    if self.print_tensors: print("read positions d")
    if self.print_tensors: print(read_positions)

    # expand to get all the w dimension locations
    visible_memory = memory.gather(1, read_positions.unsqueeze(2).expand(b, self.c, w))

    # take the vectors of the sparse reads and lru and let the read heads each look for the most similiar vector, then do softmax among all the vectors
    # for each head (b x r x (r*k + lru))
    # output shape (b x r x m), where m = r * K + 1
    
    read_weights = σ(θ(visible_memory, keys), 2)
    # let each head return one vector based on the previous softmax (b x r x w)
    read_vectors = T.bmm(read_weights, visible_memory)
    # collapses all heads into one average
    # (b x r x m) -> (b x m), where each element of m is the value of all read heads multiplied. This represents the average reading of an position

    read_weights = T.prod(read_weights, 1)


    return read_vectors, read_positions, read_weights, visible_memory

  def read(self, read_query, hidden):
    # sparse read
    read_vectors, positions, read_weights, visible_memory = \
        self.read_from_sparse_memory(
            hidden['memory'],
            hidden['indexes'],
            read_query,
            hidden['least_used_mem'],
            hidden['usage']
        )

    hidden['read_positions'] = positions
    #  use position = [2, 8 ,10] to put these sparse read location with their real read weights = [0,0,0.34,0,0,0,0,0,0,0.55,0,0.99]
    # updates the read weights only sparsely
    hidden['read_weights'] = hidden['read_weights'].scatter_(1, positions, read_weights)
    # what we actually output
    hidden['read_vectors'] = read_vectors
    hidden['visible_memory'] = visible_memory

    return hidden['read_vectors'], hidden

  def forward(self, ξ, hidden):
    t = time.time()

    # ξ = ξ.detach()
    m = self.mem_size
    w = self.cell_size
    r = self.read_heads
    c = self.c
    b = ξ.size()[0]

    if self.independent_linears:
      # r read keys (b * r * w)
      read_query = self.read_query_transform(ξ).view(b, r, w)
      # write key (b * 1 * w)
      write_vector = self.write_vector_transform(ξ).view(b, 1, w)
      # write vector (b * 1 * r)
      interpolation_gate = F.sigmoid(self.interpolation_gate_transform(ξ)).view(b, c)
      # write gate (b * 1)
      write_gate = F.sigmoid(self.write_gate_transform(ξ).view(b, 1))
    else:
      ξ = self.interface_weights(ξ)
      # r read keys (b * r * w)
      read_query = ξ[:, :r * w].contiguous().view(b, r, w)
      # write key (b * 1 * w)
      if self.direct_write:
        write_vector = ξ
        # write vector (b * 1 * r)
        interpolation_gate = F.sigmoid(ξ[:, r * w: r * w + c]).contiguous().view(b, c)
        # write gate (b * 1)
        write_gate = F.sigmoid(ξ[:, -1].contiguous()).unsqueeze(1).view(b, 1)
      else:
        write_vector = ξ[:, r * w: r * w + w].contiguous().view(b, 1, w)
        # write vector (b * 1 * r)
        interpolation_gate = F.sigmoid(ξ[:, r * w + w: r * w + w + c]).contiguous().view(b, c)
        # write gate (b * 1)
        write_gate = F.sigmoid(ξ[:, -1].contiguous()).unsqueeze(1).view(b, 1)

    self.timestep += 1
    #x changed order to first read then write
    read_vectors, hidden = self.read(read_query, hidden)
    hidden = self.write(interpolation_gate, write_vector, write_gate, hidden)
    return read_vectors, hidden
    # hidden = self.write(interpolation_gate, write_vector, write_gate, hidden)
    # return self.read(read_query, hidden)
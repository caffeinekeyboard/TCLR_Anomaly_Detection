import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pairwise_cosine_similarity(a, b):
    a_mag = torch.sqrt(torch.sum(a*a, dim=1, keepdim=True))
    b_mag = torch.sqrt(torch.sum(b*b, dim=1, keepdim=True))
    product_matrix = torch.matmul(a, b.t())
    a_mag = torch.squeeze(a_mag)
    b_mag = torch.squeeze(b_mag)
    magnitude_matrix = torch.outer(a_mag, b_mag)
    cosine_similarity_matirx = product_matrix/magnitude_matrix
    return cosine_similarity_matirx

class PairwiseEuclideanLoss(torch.nn.Module):
  def __init__(self):
    super(PairwiseEuclideanLoss, self).__init__()

  def forward(self, embeddings):
    """
      Input: A torch tensor consisting of a number of embeddings placed sequentially.
      Output: Negative cosine similarity of the embeddings.
    """
    
    pairwise_distances = torch.pdist(embeddings, p=2) ** 2 
    loss = torch.mean(pairwise_distances)  
    return loss
  


class NegativePairwiseEuclideanLossLoss(torch.nn.Module):
  def __init__(self):
    super(NegativePairwiseEuclideanLossLoss, self).__init__()

  def forward(self, embeddings):
    """
      Input: A torch tensor consisting of a number of embeddings placed sequentially.
      Output: Negative mean squared distance of the embeddings.
    """

    pairwise_distances = torch.pdist(embeddings, p=2) ** 2 
    loss = torch.mean(pairwise_distances)  
    return -loss
  


class PairwiseInstanceDiscrimination(torch.nn.Module):
  def __init__(self):
    super(PairwiseInstanceDiscrimination, self).__init__()

  def forward(self, queue_embedding, outside_embedding, tau):
    """
      Input: Temporal Queue Embeddings and Non-temporal Queue Embeddings, and temperature.
      Output: InstDisc Loss.
    """

    numerator = pairwise_cosine_similarity(queue_embedding, queue_embedding)
    denominator = pairwise_cosine_similarity(queue_embedding, outside_embedding)
    numerator = numerator/tau
    denominator = denominator/tau
    numerator = torch.exp(numerator)
    denominator = torch.exp(denominator)
    numerator = torch.mean(numerator)
    denominator = torch.mean(denominator)
    loss = -1 * torch.log(numerator/denominator)
    return loss


    
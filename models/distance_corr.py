import torch
import torch.nn as nn
import numpy as np

# class DistanceCorrelationFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, a, b, N):
#         dist_cov2_ab = torch.clamp(torch.div(torch.sum(a * b), N * N), 1e-10, 1e10)
#         dist_cov2_aa = torch.clamp(torch.div(torch.sum(a * a), N * N), 1e-10, 1e10)
#         dist_cov2_bb = torch.clamp(torch.div(torch.sum(b * b), N * N), 1e-10, 1e10)
#         dist_var_prod= torch.clamp(torch.sqrt(dist_cov2_aa) * torch.sqrt(dist_cov2_bb), 1e-10, 1e10)
#         dist_cor = torch.div(torch.sqrt(dist_cov2_ab), torch.sqrt(dist_var_prod))
        
#         return dist_cor

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output


# class DistanceCorrelation(nn.Module):
#     def __init__(self, N):
#         super(DistanceCorrelation, self).__init__()
#         self.N = N

#     def _distance_matrix(self, x):
#         a = torch.sum(x * x, 1)
#         #a = tf.reshape(a, [-1, 1])
#         a = a.view(-1, 1)
#         dA_sq = a - 2 * torch.mm(x, x.t()) + a.t()
#         dA = torch.sqrt(torch.clamp(dA_sq, 1e-10, 1e10))

#         dA_mean_row = torch.mean(dA, 0, True)
#         dA_mean_column = torch.mean(dA, 1, True)
#         dA_total_mean = torch.mean(dA)
#         dA = dA - dA_mean_row - dA_mean_column + dA_total_mean

#         return dA

#     def forward(self, A, B):
#         a = self._distance_matrix(A)
#         b = self._distance_matrix(B)
#         return DistanceCorrelationFunction.apply(a, b, self.N)

class DistanceCorrelation(nn.Module):
    def __init__(self):
        super(DistanceCorrelation, self).__init__()

    def forward(self, A, B):
        '''
        Calculate the Distance Correlation between the two vectors. https://en.wikipedia.org/wiki/Distance_correlation
        Value of 0 implies independence. A and B can be vectors of different length.
        :param A:    vector A of shape (num_samples, sizeA)
        :param B:    vector B of shape (num_samples, sizeB)
        :return:     the distance correlation between A and B
        '''
        a = self._distance_matrix(A)
        b = self._distance_matrix(B)
        dist_cov2_ab = torch.clamp(torch.div(torch.sum(a * b), A.shape[0] * A.shape[0]), 1e-10, 1e10)
        dist_cov2_aa = torch.clamp(torch.div(torch.sum(a * a), A.shape[0] * A.shape[0]), 1e-10, 1e10)
        dist_cov2_bb = torch.clamp(torch.div(torch.sum(b * b), A.shape[0] * A.shape[0]), 1e-10, 1e10)
        dist_var_prod= torch.clamp(torch.sqrt(dist_cov2_aa) * torch.sqrt(dist_cov2_bb), 1e-10, 1e10)
        dist_cor = torch.div(torch.sqrt(dist_cov2_ab), torch.sqrt(dist_var_prod))
        return dist_cor

    def _distance_matrix(self, x):
        '''
        Input: x is a Nxd matrix
            y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x**2).sum(1).view(-1, 1)
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
        
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        dist = torch.clamp(dist, 0.0, np.inf)
        dist = torch.sqrt(torch.clamp(dist, 1e-10, 1e10))
        rows_mean = torch.mean(dist, 0, True)
        columns_mean = torch.mean(dist, 1, True)
        distance = dist - rows_mean - columns_mean + torch.mean(dist)
        return distance
        # z = torch.sum(x * x, 1)
        # #a = tf.reshape(a, [-1, 1])
        # z = z.view(-1, 1)
        # dZ_sq = z - 2 * torch.mm(x, x.t()) + z.t()
        # dZ = torch.sqrt(torch.clamp(dZ_sq, 1e-10, 1e10))

        # dZ_mean_row = torch.mean(dZ, 0, True)
        # dZ_mean_column = torch.mean(dZ, 1, True)
        # dZ_total_mean = torch.mean(dZ)
        # dZ = dZ - dZ_mean_row - dZ_mean_column + dZ_total_mean
        # return dZ

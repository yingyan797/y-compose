import torch
import torch.nn as nn
import torch.nn.functional as F

class NAFNetwork(nn.Module):
    """
    NAF architecture that outputs value, advantage, and mu for analytical argmax
    """
    def __init__(self, state_dim, goal_dim, action_dim):
        super(NAFNetwork, self).__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        input_dim = state_dim + goal_dim
        
        # Shared feature layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Value stream - V(s,g)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Action policy stream - μ(s,g)
        self.mu = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
        
        # Advantage stream parameters
        # Lower triangular matrix for Cholesky decomposition of P
        self.L_layer = nn.Linear(256, (action_dim * (action_dim + 1)) // 2)
        
        # Diagonal elements will have positive values
        self.tril_mask = torch.tril(torch.ones(action_dim, action_dim))
        self.diag_mask = torch.diag(torch.ones(action_dim))
        
    def forward(self, state, goal, action=None):
        """Forward pass that returns Q-value and optionally policy parameters"""
        x = torch.cat([state, goal], dim=1)
        shared_features = self.shared_layers(x)
        
        # State-goal value
        V = self.value_stream(shared_features)
        
        # Action mean (optimal action)
        mu = self.mu(shared_features)
        
        # Advantage function parameters
        L_vector = self.L_layer(shared_features)
        
        # Create lower triangular matrix L
        batch_size = x.shape[0]
        L = torch.zeros(batch_size, self.action_dim, self.action_dim, device=x.device)
        
        # Fill in the lower triangular part
        tril_indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = L_vector
        
        # Make diagonal elements positive with softplus
        diagonal_indices = torch.IntTensor([[i,i] for i in range(self.action_dim)]).to(device=x.device)
        L[:, diagonal_indices[:,0], diagonal_indices[:,1]] = F.softplus(L[:, diagonal_indices[:,0], diagonal_indices[:,1]])
        
        # Compute precision matrix P = L*L^T (positive definite)
        P = torch.bmm(L, L.transpose(1, 2))
        
        # If action is provided, compute Q-value
        if action is not None:
            # Advantage = -1/2 * (a - μ(s,g))^T * P * (a - μ(s,g))
            action_diff = action - mu
            quadratic_term = torch.bmm(torch.bmm(action_diff.unsqueeze(1), P), action_diff.unsqueeze(2))
            A = -0.5 * quadratic_term.squeeze()
            
            # Q = V + A
            Q = V + A
            return Q, mu, P, V
        else:
            # For optimal actions, advantage is 0, so Q = V
            return V, mu, P, V

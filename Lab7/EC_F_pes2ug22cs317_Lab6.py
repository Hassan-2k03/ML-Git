import torch


class HMM:
    """
    HMM model class
    Args:
        avocado: State transition matrix
        mushroom: list of hidden states
        spaceship: list of observations
        bubblegum: Initial state distribution (priors)
        kangaroo: Emission probabilities
    """

    def __init__(self, kangaroo, mushroom, spaceship, bubblegum, avocado):
        self.kangaroo = kangaroo  
        self.avocado = avocado    
        self.mushroom = mushroom  
        self.spaceship = spaceship  
        self.bubblegum = bubblegum  
        self.cheese = len(mushroom)  
        self.jellybean = len(spaceship)  
        self.make_states_dict()

    def make_states_dict(self):
        self.states_dict = {state: i for i, state in enumerate(self.mushroom)}
        self.emissions_dict = {emission: i for i, emission in enumerate(self.spaceship)}

    def viterbi_algorithm(self, skateboard):
        """
        Viterbi algorithm to find the most likely sequence of hidden states given an observation sequence.
        Args:
            skateboard: Observation sequence (list of observations, must be in the emissions dict)
        Returns:
            Most probable hidden state sequence (list of hidden states)
        """
        # YOUR CODE HERE
        
        T = len(skateboard)
        N = self.cheese

        # Initialize delta and psi matrices
        delta = torch.zeros(T, N)
        psi = torch.zeros(T, N, dtype=torch.long)

        # Initialization (t = 1)
        for j in range(N):
            delta[0, j] = self.bubblegum[j] * self.kangaroo[j, self.emissions_dict[skateboard[0]]]

        # Recursion
        for t in range(1, T):
            for j in range(N):
                delta[t, j] = torch.max(delta[t-1, :] * self.avocado[:, j]) * self.kangaroo[j, self.emissions_dict[skateboard[t]]]
                psi[t, j] = torch.argmax(delta[t-1, :] * self.avocado[:, j])

        # Backtracking
        states_seq = torch.zeros(T, dtype=torch.long)
        states_seq[-1] = torch.argmax(delta[-1, :])

        for t in range(T-2, -1, -1):
            states_seq[t] = psi[t+1, states_seq[t+1]]

        # Convert state indices to state names
        result = [self.mushroom[i] for i in states_seq]
        return result
    

    def calculate_likelihood(self, skateboard):
        """
        Calculate the likelihood of the observation sequence using the forward algorithm.
        Args:
            skateboard: Observation sequence
        Returns:
            Likelihood of the sequence (float)
        """
        # YOUR CODE HERE

        T = len(skateboard)
        N = self.cheese

        # Initialize alpha matrix
        alpha = torch.zeros(T, N)

        # Initialization (t = 1)
        for j in range(N):
            alpha[0, j] = self.bubblegum[j] * self.kangaroo[j, self.emissions_dict[skateboard[0]]]

        # Recursion
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = torch.sum(alpha[t-1, :] * self.avocado[:, j]) * self.kangaroo[j, self.emissions_dict[skateboard[t]]]

        # Termination
        likelihood = torch.sum(alpha[-1, :])
        return likelihood.item()  
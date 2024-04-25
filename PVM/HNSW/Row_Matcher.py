import hnswlib
import pandas as pd
import numpy as np
import torch
import statsmodels.api as sm

from statsmodels.discrete.discrete_model import Probit
from statsmodels.api import OLS

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class RowMatcher:
    def create_index(self, model, dataframe, device='cpu'):
        """Create an HNSW index from a DataFrame using embeddings from a model."""
        model.eval()
        embeddings = []

        model.to(device)

        with torch.no_grad():
            # Ensure unique_indices is a flat list of integers
            unique_indices = np.unique(dataframe['encoded_vector'].to_numpy())
            for index in unique_indices:
                tensor_index = torch.tensor([index], dtype=torch.long, device=device)
                embedding = model.vq_vae.embedding(tensor_index).squeeze(0).cpu().numpy()
                embeddings.append(embedding)

        embeddings = np.vstack(embeddings)

        # Initialize and fill the HNSW index
        dim = embeddings.shape[1]
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
        index.add_items(embeddings, np.arange(len(embeddings)))

        return index
    
    def retrieve_similar(self, df_randhie, df_heart, randhie_model, heart_index, device='cpu'):
        """Retrieve similar rows based on HNSW index."""
        randhie_model.eval()
        randhie_embeddings = []
        
        randhie_model.to(device)
        
        with torch.no_grad():
            # Generate embeddings for the randhie data
            for idx in df_randhie['encoded_vector']:
                tensor_index = torch.tensor([idx], dtype=torch.long, device=device)
                embedding = randhie_model.vq_vae.embedding(tensor_index).squeeze(0).cpu().numpy()
                randhie_embeddings.append(embedding)
        
        randhie_embeddings = np.array(randhie_embeddings)
        
        # Query the HNSW index for nearest neighbors
        labels, distances = heart_index.knn_query(randhie_embeddings, k=1)
        
        # Fetch the most similar rows from the heart dataframe
        similar_rows = df_heart.iloc[labels.flatten()]
        
        # Combine the data for display or further analysis
        combined_data = df_randhie.copy()
        combined_data['matched_index'] = labels.flatten()
        combined_data['distance'] = distances.flatten()
        combined_data['matched_row'] = similar_rows.reset_index(drop=True)
        
        return combined_data
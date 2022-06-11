"""Importing dependencies"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd


class PCA():
    """
    This class performs Principle Component Analysis (PCA) on scaled data"""

    def __init__(self,X):
        """
        Initialaze
        """

        self.X = X
        self.covariance_matrix = self.cov()
        self.vals = self.eig()[0]
        self.vecs = self.eig()[1]
        self.projections = self.get_projections()

    def get_vals(self):
        """
        Returns:
                Eigenvalues
        """
        return self.vals

    def get_vecs(self):
        """
        Returns:
                Eigenvectors
        """
        return self.vecs


    def cov(self):
        """
        arr: self.X

        Returns:
                Covariance matrix
        """
        self.covariance_matrix = np.dot(self.X.T, self.X) / (self.X.shape[0] - 1)
        return self.covariance_matrix


    def eig(self):
        """
        arr:
                self.covariance_matrix

        returns:
                sorted eigenvalues highest to lowest
                sortrd eigenvectors by eigenvalues
        """
        self.vals, self.vecs = np.linalg.eig(self.covariance_matrix)

        # Sort eigenvectors and values by eigenvalues
        order = np.argsort(self.vals)[::-1]
        self.vals =self. vals[order]
        self.vecs = self.vecs[:, order]

        return self.vals, self.vecs

    def scree_plot(self):
        """
        arr:
                self.vals
                self.vecs

        returns:
                scree plot
        """
        # Scree plot  -> Kaiser rule:
        plt.bar(np.arange(len(self.vals)) + 1, self.vals, label = 'PC')

        plt.axhline(1, label = 'eigenvalues > 1')
        plt.title("Kaiser rule")
        plt.ylabel('Eigenvalues')
        plt.xlabel('No. of components')
        plt.legend()
        plt.show()

    def get_projections(self):
        """
        arr:
                self.X
                self.vecs
                self.vals

        returns:
               self.projections as pandas dataframe
        """

        pca_dict= {}
        for n_vec in range(len(self.vecs)):

            pca_dict['PC{}'.format(n_vec + 1)] = np.dot(self.X,self.vecs[:, n_vec])
            self.projections = pd.DataFrame(pca_dict)

        return self.projections

    def plot_PCA(self, x = 'PC1', y = 'PC2', color = 'PC3'):
        """
        arr:
                self.projections
        returns:
                PCA plot
        """

        fig = px.scatter(self.projections, x = x, y=y, color = color)

        fig.update_layout(
            width = 600,
            height = 600,
            title = "Priciple Component Analysis"
        )
        fig.show()

def main():
    """
    main function
    """
    print('PCA module')

if __name__ == "__main__":
    main()

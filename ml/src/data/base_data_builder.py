"""
Rôle : Définir une interface abstraite et stricte (un "contrat") pour la préparation des données. 
Quel que soit le framework Machine Learning utilisé (YOLO, PyTorch, TensorFlow), 
le composant chargé des données devra implémenter ce comportement.

"""

from abc import ABC, abstractmethod

class BaseDataBuilder(ABC):
    """
    Interface abstraite pour la préparation des données.
    """
    @abstractmethod
    def prepare_data(self) -> str:
        """
        Gère la récupération (locale ou distante) et le formatage des données.
        Retourne le chemin absolu vers le dossier ou le fichier de configuration 
        des données prêt à être consommé par le modèle.
        """
        pass
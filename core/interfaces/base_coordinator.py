from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class NetworkCoordinator(ABC):
    """
    Abstract base class for communicating with arbitrary distributed computing managers
    or databases (Firebase, SQLite, Kafka, etc.) to negotiate work payloads.
    """
    @abstractmethod
    def fetch_work_unit(self) -> Optional[Dict]:
        """
        Retrieves a continuous bounding phase-space block to securely process.
        
        Returns:
            A dictionary containing the bounds payload, or None if the search space is exhausted.
        """
        pass

    @abstractmethod
    def submit_results(self, verified_discoveries: List[Dict]) -> bool:
        """
        Transmits mathematically verified Generalized Continued Fractions structurally
        back to the centralized oracle / storage persistence layer.
        
        Returns:
            True if successful. On False, the framework triggers local caching overrides.
        """
        pass

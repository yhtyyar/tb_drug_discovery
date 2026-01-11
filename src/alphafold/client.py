"""AlphaFold Server API client.

This module provides a client for interacting with AlphaFold Server
for structure prediction of proteins and protein-ligand complexes.

Supports:
- AlphaFold Server (web API)
- Local ColabFold installation
- Structure caching

Example:
    >>> client = AlphaFoldClient()
    >>> result = client.predict_structure(sequence="MKFLILLFNILC...")
    >>> result.save_pdb("output.pdb")
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from loguru import logger

try:
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Structure import Structure
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    logger.warning("BioPython not installed. Some features will be limited.")


@dataclass
class AlphaFoldConfig:
    """Configuration for AlphaFold client.
    
    Args:
        server_url: AlphaFold Server URL.
        api_key: Optional API key for authentication.
        cache_dir: Directory for caching results.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
        use_cache: Whether to use cached results.
    """
    server_url: str = "https://alphafoldserver.com/api"
    api_key: Optional[str] = None
    cache_dir: str = "data/alphafold_cache"
    timeout: int = 300
    max_retries: int = 3
    use_cache: bool = True
    poll_interval: int = 30  # seconds between status checks


@dataclass
class PredictionResult:
    """Result from AlphaFold structure prediction.
    
    Args:
        job_id: Unique job identifier.
        sequence: Input protein sequence.
        pdb_string: Predicted structure in PDB format.
        plddt_scores: Per-residue confidence scores.
        pae_matrix: Predicted aligned error matrix.
        metadata: Additional prediction metadata.
    """
    job_id: str
    sequence: str
    pdb_string: str
    plddt_scores: List[float] = field(default_factory=list)
    pae_matrix: Optional[List[List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save_pdb(self, path: str) -> None:
        """Save structure to PDB file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.pdb_string)
        logger.info(f"Structure saved to {path}")
    
    def get_mean_plddt(self) -> float:
        """Get mean pLDDT confidence score."""
        if not self.plddt_scores:
            return 0.0
        return sum(self.plddt_scores) / len(self.plddt_scores)
    
    def get_confident_residues(self, threshold: float = 70.0) -> List[int]:
        """Get indices of residues with high confidence."""
        return [i for i, score in enumerate(self.plddt_scores) if score >= threshold]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "sequence": self.sequence,
            "pdb_string": self.pdb_string,
            "plddt_scores": self.plddt_scores,
            "pae_matrix": self.pae_matrix,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionResult":
        """Create from dictionary."""
        return cls(**data)


class AlphaFoldClient:
    """Client for AlphaFold structure prediction.
    
    Provides interface to AlphaFold Server for predicting protein
    structures and protein-ligand complexes.
    
    Args:
        config: AlphaFoldConfig instance.
        
    Example:
        >>> client = AlphaFoldClient()
        >>> # Single protein
        >>> result = client.predict_structure("MKFLILLFNILC...")
        >>> # Protein-ligand complex
        >>> result = client.predict_complex("MKFLILLFNILC...", "CCO")
    """
    
    def __init__(self, config: Optional[AlphaFoldConfig] = None):
        self.config = config or AlphaFoldConfig()
        
        # Setup cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Session for HTTP requests
        self.session = requests.Session()
        if self.config.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        logger.info(f"AlphaFoldClient initialized. Cache: {self.cache_dir}")
    
    def predict_structure(
        self,
        sequence: str,
        name: str = "protein",
        use_templates: bool = True,
    ) -> PredictionResult:
        """Predict protein structure from sequence.
        
        Args:
            sequence: Amino acid sequence (single letter code).
            name: Name for the prediction job.
            use_templates: Use template-based modeling.
            
        Returns:
            PredictionResult with predicted structure.
        """
        # Check cache
        cache_key = self._get_cache_key(sequence, "structure")
        if self.config.use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                logger.info(f"Loaded structure from cache: {cache_key}")
                return cached
        
        # Validate sequence
        sequence = self._clean_sequence(sequence)
        
        # Submit job
        job_id = self._submit_structure_job(sequence, name, use_templates)
        
        # Wait for completion
        result = self._wait_for_result(job_id)
        
        # Cache result
        if self.config.use_cache:
            self._save_to_cache(cache_key, result)
        
        return result
    
    def predict_complex(
        self,
        protein_sequence: str,
        ligand_smiles: str,
        name: str = "complex",
    ) -> PredictionResult:
        """Predict protein-ligand complex structure.
        
        Args:
            protein_sequence: Protein amino acid sequence.
            ligand_smiles: Ligand SMILES string.
            name: Name for the prediction job.
            
        Returns:
            PredictionResult with predicted complex structure.
        """
        # Check cache
        cache_key = self._get_cache_key(f"{protein_sequence}_{ligand_smiles}", "complex")
        if self.config.use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                logger.info(f"Loaded complex from cache: {cache_key}")
                return cached
        
        # Clean inputs
        protein_sequence = self._clean_sequence(protein_sequence)
        
        # Submit job
        job_id = self._submit_complex_job(protein_sequence, ligand_smiles, name)
        
        # Wait for completion
        result = self._wait_for_result(job_id)
        
        # Cache result
        if self.config.use_cache:
            self._save_to_cache(cache_key, result)
        
        return result
    
    def predict_multimer(
        self,
        sequences: List[str],
        chains: Optional[List[str]] = None,
        name: str = "multimer",
    ) -> PredictionResult:
        """Predict multi-chain protein complex.
        
        Args:
            sequences: List of protein sequences.
            chains: Optional chain identifiers.
            name: Name for the prediction job.
            
        Returns:
            PredictionResult with predicted multimer structure.
        """
        if chains is None:
            chains = [chr(65 + i) for i in range(len(sequences))]  # A, B, C, ...
        
        # Check cache
        cache_key = self._get_cache_key("_".join(sequences), "multimer")
        if self.config.use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # Clean sequences
        sequences = [self._clean_sequence(seq) for seq in sequences]
        
        # Submit job
        job_id = self._submit_multimer_job(sequences, chains, name)
        
        # Wait for completion
        result = self._wait_for_result(job_id)
        
        # Cache result
        if self.config.use_cache:
            self._save_to_cache(cache_key, result)
        
        return result
    
    def _submit_structure_job(
        self,
        sequence: str,
        name: str,
        use_templates: bool,
    ) -> str:
        """Submit structure prediction job."""
        payload = {
            "name": name,
            "sequences": [
                {"proteinChain": {"sequence": sequence, "count": 1}}
            ],
            "modelSeeds": [1],
        }
        
        return self._submit_job(payload)
    
    def _submit_complex_job(
        self,
        protein_sequence: str,
        ligand_smiles: str,
        name: str,
    ) -> str:
        """Submit protein-ligand complex prediction job."""
        payload = {
            "name": name,
            "sequences": [
                {"proteinChain": {"sequence": protein_sequence, "count": 1}},
                {"ligand": {"smiles": ligand_smiles, "count": 1}},
            ],
            "modelSeeds": [1],
        }
        
        return self._submit_job(payload)
    
    def _submit_multimer_job(
        self,
        sequences: List[str],
        chains: List[str],
        name: str,
    ) -> str:
        """Submit multimer prediction job."""
        payload = {
            "name": name,
            "sequences": [
                {"proteinChain": {"sequence": seq, "count": 1}}
                for seq in sequences
            ],
            "modelSeeds": [1],
        }
        
        return self._submit_job(payload)
    
    def _submit_job(self, payload: Dict[str, Any]) -> str:
        """Submit job to AlphaFold Server."""
        url = f"{self.config.server_url}/jobs"
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                
                result = response.json()
                job_id = result.get("jobId", result.get("id"))
                
                logger.info(f"Submitted job: {job_id}")
                return job_id
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise RuntimeError(f"Failed to submit job after {self.config.max_retries} attempts") from e
        
        raise RuntimeError("Failed to submit job")
    
    def _wait_for_result(self, job_id: str) -> PredictionResult:
        """Wait for job completion and retrieve result."""
        url = f"{self.config.server_url}/jobs/{job_id}"
        
        while True:
            try:
                response = self.session.get(url, timeout=self.config.timeout)
                response.raise_for_status()
                
                result = response.json()
                status = result.get("state", result.get("status", "UNKNOWN"))
                
                if status == "SUCCEEDED":
                    return self._parse_result(job_id, result)
                elif status in ["FAILED", "CANCELLED"]:
                    raise RuntimeError(f"Job {job_id} failed with status: {status}")
                else:
                    logger.debug(f"Job {job_id} status: {status}")
                    time.sleep(self.config.poll_interval)
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error checking job status: {e}")
                time.sleep(self.config.poll_interval)
    
    def _parse_result(self, job_id: str, result: Dict[str, Any]) -> PredictionResult:
        """Parse prediction result from API response."""
        # Extract PDB string
        pdb_url = result.get("pdbUrl", result.get("structureUrl"))
        if pdb_url:
            pdb_response = self.session.get(pdb_url)
            pdb_string = pdb_response.text
        else:
            pdb_string = result.get("pdbString", "")
        
        # Extract confidence scores
        plddt_scores = result.get("plddtScores", [])
        pae_matrix = result.get("paeMatrix")
        
        # Extract sequence
        sequences = result.get("sequences", [])
        sequence = ""
        if sequences:
            first_seq = sequences[0]
            if isinstance(first_seq, dict):
                protein_chain = first_seq.get("proteinChain", {})
                sequence = protein_chain.get("sequence", "")
            else:
                sequence = first_seq
        
        return PredictionResult(
            job_id=job_id,
            sequence=sequence,
            pdb_string=pdb_string,
            plddt_scores=plddt_scores,
            pae_matrix=pae_matrix,
            metadata=result,
        )
    
    def _clean_sequence(self, sequence: str) -> str:
        """Clean and validate protein sequence."""
        # Remove whitespace and convert to uppercase
        sequence = "".join(sequence.split()).upper()
        
        # Validate amino acids
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = set(sequence) - valid_aa
        if invalid:
            logger.warning(f"Removing invalid characters: {invalid}")
            sequence = "".join(c for c in sequence if c in valid_aa)
        
        return sequence
    
    def _get_cache_key(self, content: str, prefix: str) -> str:
        """Generate cache key from content."""
        hash_value = hashlib.md5(content.encode()).hexdigest()[:16]
        return f"{prefix}_{hash_value}"
    
    def _save_to_cache(self, key: str, result: PredictionResult) -> None:
        """Save result to cache."""
        cache_path = self.cache_dir / f"{key}.json"
        with open(cache_path, "w") as f:
            json.dump(result.to_dict(), f)
        
        # Also save PDB file
        pdb_path = self.cache_dir / f"{key}.pdb"
        with open(pdb_path, "w") as f:
            f.write(result.pdb_string)
        
        logger.debug(f"Cached result: {key}")
    
    def _load_from_cache(self, key: str) -> Optional[PredictionResult]:
        """Load result from cache."""
        cache_path = self.cache_dir / f"{key}.json"
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            return PredictionResult.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        import shutil
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cache cleared")
    
    def get_cached_results(self) -> List[str]:
        """List all cached result keys."""
        return [p.stem for p in self.cache_dir.glob("*.json")]


class LocalAlphaFoldClient(AlphaFoldClient):
    """Client for local AlphaFold/ColabFold installation.
    
    For running predictions on local GPU infrastructure.
    
    Args:
        colabfold_path: Path to ColabFold installation.
        config: AlphaFoldConfig instance.
    """
    
    def __init__(
        self,
        colabfold_path: Optional[str] = None,
        config: Optional[AlphaFoldConfig] = None,
    ):
        super().__init__(config)
        self.colabfold_path = colabfold_path
        
        # Check if ColabFold is available
        self.colabfold_available = self._check_colabfold()
    
    def _check_colabfold(self) -> bool:
        """Check if ColabFold is available."""
        try:
            import subprocess
            result = subprocess.run(
                ["colabfold_batch", "--help"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def predict_structure(
        self,
        sequence: str,
        name: str = "protein",
        use_templates: bool = True,
        num_models: int = 1,
    ) -> PredictionResult:
        """Predict structure using local ColabFold."""
        if not self.colabfold_available:
            logger.warning("ColabFold not available, falling back to server")
            return super().predict_structure(sequence, name, use_templates)
        
        # Check cache
        cache_key = self._get_cache_key(sequence, "local_structure")
        if self.config.use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # Run ColabFold locally
        result = self._run_colabfold(sequence, name, num_models)
        
        # Cache result
        if self.config.use_cache:
            self._save_to_cache(cache_key, result)
        
        return result
    
    def _run_colabfold(
        self,
        sequence: str,
        name: str,
        num_models: int,
    ) -> PredictionResult:
        """Run ColabFold prediction locally."""
        import subprocess
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write FASTA file
            fasta_path = os.path.join(tmpdir, "input.fasta")
            with open(fasta_path, "w") as f:
                f.write(f">{name}\n{sequence}\n")
            
            # Run ColabFold
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir)
            
            cmd = [
                "colabfold_batch",
                fasta_path,
                output_dir,
                "--num-models", str(num_models),
                "--amber",
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"ColabFold failed: {result.stderr}")
            
            # Find output PDB
            pdb_files = list(Path(output_dir).glob("*.pdb"))
            if not pdb_files:
                raise RuntimeError("No PDB output found")
            
            pdb_path = pdb_files[0]
            with open(pdb_path, "r") as f:
                pdb_string = f.read()
            
            # Parse scores from JSON
            scores_path = Path(output_dir) / f"{name}_scores.json"
            plddt_scores = []
            if scores_path.exists():
                with open(scores_path, "r") as f:
                    scores = json.load(f)
                    plddt_scores = scores.get("plddt", [])
            
            return PredictionResult(
                job_id=f"local_{name}",
                sequence=sequence,
                pdb_string=pdb_string,
                plddt_scores=plddt_scores,
                metadata={"local": True},
            )

import json
import random
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from server.base import MCPServer


class TransformerMCPServer(MCPServer):
    """
    MCP Server for Transformer-based EHR and Lab Test inference.
    
    This server handles:
    - EHR text encoding and processing
    - Lab test numerical feature extraction
    - Multi-modal fusion for prediction
    - Risk score calculation
    """
    
    SERVER_KEY = "transformer"
    
    def __init__(
        self,
        model_name: str = "transformer-ehr-v1",
        model_path: Optional[str] = None,
        batch_size: int = 32,
        max_sequence_length: int = 512,
        device: str = "cpu",
        **kwargs
    ) -> None:
        """Initialize the Transformer MCP Server."""
        self.model_name = model_name
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Default Redis keys for transformer server
        task_queue_key = kwargs.pop("task_queue_key", "mochiagent:transformer:task")
        result_queue_key = kwargs.pop("result_queue_key", "mochiagent:transformer:result")
        
        super().__init__(
            task_queue_key=task_queue_key,
            result_queue_key=result_queue_key,
            **kwargs
        )
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the transformer model and tokenizer."""
        self._log_info(f"Initializing transformer model: {self.model_name}")
        # In production, load actual model here
        # self.model = AutoModel.from_pretrained(self.model_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._log_info("Model initialization complete (placeholder mode)")
    
    def _register_tools(self) -> None:
        """Register transformer-specific tools."""
        self.register_tool(
            "predict",
            self._tool_predict,
            "Perform prediction on EHR and lab test data"
        )
        self.register_tool(
            "encode_ehr",
            self._tool_encode_ehr,
            "Encode EHR text data into embeddings"
        )
        self.register_tool(
            "encode_lab_tests",
            self._tool_encode_lab_tests,
            "Encode lab test numerical data"
        )
        self.register_tool(
            "risk_assessment",
            self._tool_risk_assessment,
            "Perform comprehensive risk assessment"
        )
        self.register_tool(
            "attention_analysis",
            self._tool_attention_analysis,
            "Analyze attention patterns for interpretability"
        )
    
    def _register_resources(self) -> None:
        """Register transformer-specific resources."""
        self.register_resource(
            "model://transformer/config",
            {
                "model_name": self.model_name,
                "max_sequence_length": self.max_sequence_length,
                "batch_size": self.batch_size,
                "device": self.device
            }
        )
        self.register_resource(
            "model://transformer/vocabulary",
            self._get_vocabulary_info()
        )
    
    def _register_prompts(self) -> None:
        """Register transformer-specific prompts."""
        self.register_prompt(
            "ehr_analysis",
            "Analyze the following EHR record and identify key clinical features:\n{ehr_text}"
        )
        self.register_prompt(
            "risk_explanation",
            "Based on the prediction score of {score}, explain the risk factors:\n{factors}"
        )
    
    def _get_vocabulary_info(self) -> Dict[str, Any]:
        """Get vocabulary information."""
        return {
            "vocab_size": 30522,
            "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            "max_length": self.max_sequence_length
        }
    
    def _preprocess_ehr(self, ehr_list: List[str]) -> Dict[str, Any]:
        """Preprocess EHR text data."""
        processed = []
        for ehr in ehr_list:
            # Normalize text
            text = ehr.strip().lower()
            # Truncate to max length
            if len(text) > self.max_sequence_length * 4:
                text = text[:self.max_sequence_length * 4]
            processed.append(text)
        
        return {
            "texts": processed,
            "count": len(processed),
            "avg_length": sum(len(t) for t in processed) / len(processed) if processed else 0
        }
    
    def _preprocess_lab_tests(self, lab_tests: List[List[Any]]) -> Dict[str, Any]:
        """Preprocess lab test numerical data."""
        # Convert to numpy array for numerical processing
        try:
            array = np.array(lab_tests, dtype=np.float32)
        except (ValueError, TypeError):
            # Handle mixed types
            array = np.array([[float(v) if v is not None else 0.0 for v in row] for row in lab_tests])
        
        # Normalize features
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        normalized = (array - mean) / std
        
        return {
            "original": array.tolist(),
            "normalized": normalized.tolist(),
            "statistics": {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "min": np.min(array, axis=0).tolist(),
                "max": np.max(array, axis=0).tolist()
            },
            "shape": list(array.shape)
        }
    
    def _compute_embeddings(self, preprocessed_ehr: Dict, preprocessed_lab: Dict) -> Dict[str, Any]:
        """Compute embeddings for EHR and lab test data."""
        # Placeholder for actual embedding computation
        # In production, this would use the transformer model
        
        ehr_embedding_dim = 768
        lab_embedding_dim = 256
        
        num_ehr = preprocessed_ehr["count"]
        num_lab_rows = len(preprocessed_lab["normalized"])
        
        # Generate placeholder embeddings
        ehr_embeddings = np.random.randn(num_ehr, ehr_embedding_dim).tolist()
        lab_embeddings = np.random.randn(num_lab_rows, lab_embedding_dim).tolist()
        
        # Fused embedding (concatenation + projection)
        fused_dim = 512
        fused_embedding = np.random.randn(fused_dim).tolist()
        
        return {
            "ehr_embeddings": ehr_embeddings,
            "lab_embeddings": lab_embeddings,
            "fused_embedding": fused_embedding,
            "dimensions": {
                "ehr": ehr_embedding_dim,
                "lab": lab_embedding_dim,
                "fused": fused_dim
            }
        }
    
    def _compute_prediction(self, embeddings: Dict[str, Any]) -> Dict[str, Any]:
        """Compute prediction from embeddings."""
        # Placeholder prediction logic
        # In production, this would use a classification/regression head
        
        fused = embeddings["fused_embedding"]
        
        # Simulate prediction computation
        prediction_score = random.random()
        confidence = 0.7 + random.random() * 0.25
        
        # Risk categorization
        if prediction_score < 0.3:
            risk_category = "LOW"
        elif prediction_score < 0.7:
            risk_category = "MODERATE"
        else:
            risk_category = "HIGH"
        
        return {
            "prediction_score": prediction_score,
            "confidence": confidence,
            "risk_category": risk_category,
            "probabilities": {
                "low_risk": max(0, 1 - prediction_score - 0.1 + random.random() * 0.1),
                "moderate_risk": 0.3 + random.random() * 0.2,
                "high_risk": prediction_score * 0.8
            }
        }
    
    def _tool_predict(
        self,
        ehr_data: List[str],
        lab_tests: List[List[Any]],
        return_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        Main prediction tool.
        
        Args:
            ehr_data: List of EHR text records
            lab_tests: 2D list of lab test values
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Prediction results including score, confidence, and risk category
        """
        self._log_info(f"Processing prediction request: {len(ehr_data)} EHR records, {len(lab_tests)} lab test rows")
        
        # Preprocessing
        preprocessed_ehr = self._preprocess_ehr(ehr_data)
        preprocessed_lab = self._preprocess_lab_tests(lab_tests)
        
        # Compute embeddings
        embeddings = self._compute_embeddings(preprocessed_ehr, preprocessed_lab)
        
        # Compute prediction
        prediction = self._compute_prediction(embeddings)
        
        result = {
            "status": "success",
            "prediction": prediction,
            "preprocessing_info": {
                "ehr_count": preprocessed_ehr["count"],
                "ehr_avg_length": preprocessed_ehr["avg_length"],
                "lab_shape": preprocessed_lab["shape"]
            },
            "model_info": {
                "name": self.model_name,
                "device": self.device
            },
            "timestamp": datetime.now().isoformat()
        }
        
        if return_embeddings:
            result["embeddings"] = embeddings
        
        return result
    
    def _tool_encode_ehr(self, ehr_data: List[str]) -> Dict[str, Any]:
        """Encode EHR text data into embeddings."""
        preprocessed = self._preprocess_ehr(ehr_data)
        
        # Generate embeddings
        embedding_dim = 768
        embeddings = np.random.randn(preprocessed["count"], embedding_dim).tolist()
        
        return {
            "embeddings": embeddings,
            "dimension": embedding_dim,
            "count": preprocessed["count"],
            "model": self.model_name
        }
    
    def _tool_encode_lab_tests(self, lab_tests: List[List[Any]]) -> Dict[str, Any]:
        """Encode lab test numerical data."""
        preprocessed = self._preprocess_lab_tests(lab_tests)
        
        embedding_dim = 256
        embeddings = np.random.randn(len(lab_tests), embedding_dim).tolist()
        
        return {
            "embeddings": embeddings,
            "dimension": embedding_dim,
            "statistics": preprocessed["statistics"],
            "shape": preprocessed["shape"]
        }
    
    def _tool_risk_assessment(
        self,
        ehr_data: List[str],
        lab_tests: List[List[Any]],
        assessment_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment.
        
        Args:
            ehr_data: List of EHR text records
            lab_tests: 2D list of lab test values
            assessment_type: Type of assessment (basic, comprehensive, detailed)
            
        Returns:
            Risk assessment results
        """
        # Get base prediction
        prediction = self._tool_predict(ehr_data, lab_tests)
        
        # Compute additional risk factors
        risk_factors = []
        
        # Analyze lab test anomalies
        preprocessed_lab = self._preprocess_lab_tests(lab_tests)
        for i, (mean, std) in enumerate(zip(
            preprocessed_lab["statistics"]["mean"],
            preprocessed_lab["statistics"]["std"]
        )):
            if abs(mean) > 2 * std:
                risk_factors.append({
                    "factor": f"lab_test_{i}",
                    "type": "abnormal_value",
                    "severity": "moderate" if abs(mean) < 3 * std else "high"
                })
        
        # Compute temporal trends if multiple time points
        if len(lab_tests) > 1:
            trends = []
            array = np.array(preprocessed_lab["normalized"])
            for col in range(array.shape[1]):
                slope = np.polyfit(range(len(array)), array[:, col], 1)[0]
                if abs(slope) > 0.5:
                    trends.append({
                        "feature_index": col,
                        "trend": "increasing" if slope > 0 else "decreasing",
                        "magnitude": abs(slope)
                    })
            if trends:
                risk_factors.append({
                    "factor": "temporal_trends",
                    "type": "trend_analysis",
                    "details": trends
                })
        
        return {
            "assessment_type": assessment_type,
            "overall_risk": prediction["prediction"]["risk_category"],
            "risk_score": prediction["prediction"]["prediction_score"],
            "confidence": prediction["prediction"]["confidence"],
            "risk_factors": risk_factors,
            "recommendations": self._generate_recommendations(
                prediction["prediction"]["risk_category"],
                risk_factors
            ),
            "timestamp": datetime.now().isoformat()
        }
    
    def _tool_attention_analysis(
        self,
        ehr_data: List[str],
        lab_tests: List[List[Any]]
    ) -> Dict[str, Any]:
        """Analyze attention patterns for model interpretability."""
        num_ehr = len(ehr_data)
        num_lab = len(lab_tests)
        
        # Generate placeholder attention weights
        # In production, these would come from the transformer model
        
        # Self-attention within EHR
        ehr_self_attention = np.random.rand(num_ehr, num_ehr).tolist()
        
        # Cross-attention between EHR and lab tests
        cross_attention = np.random.rand(num_ehr, num_lab).tolist()
        
        # Feature importance scores
        lab_importance = np.random.rand(len(lab_tests[0]) if lab_tests else 0).tolist()
        
        return {
            "ehr_self_attention": ehr_self_attention,
            "cross_attention": cross_attention,
            "lab_feature_importance": lab_importance,
            "interpretation": {
                "most_important_ehr": 0 if num_ehr > 0 else None,
                "most_important_lab_feature": np.argmax(lab_importance) if lab_importance else None,
                "attention_entropy": float(np.mean([
                    -np.sum(row * np.log(row + 1e-10))
                    for row in np.array(ehr_self_attention)
                ])) if ehr_self_attention else 0
            }
        }
    
    def _generate_recommendations(
        self,
        risk_category: str,
        risk_factors: List[Dict]
    ) -> List[str]:
        """Generate recommendations based on risk assessment."""
        recommendations = []
        
        if risk_category == "HIGH":
            recommendations.append("Immediate clinical review recommended")
            recommendations.append("Consider additional diagnostic tests")
        elif risk_category == "MODERATE":
            recommendations.append("Schedule follow-up within 2 weeks")
            recommendations.append("Monitor key indicators")
        else:
            recommendations.append("Continue routine monitoring")
        
        # Add factor-specific recommendations
        for factor in risk_factors:
            if factor["type"] == "abnormal_value":
                recommendations.append(f"Review {factor['factor']} values")
            elif factor["type"] == "trend_analysis":
                recommendations.append("Evaluate temporal patterns in lab results")
        
        return recommendations
    
    def _handle_custom_method(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom methods for transformer server."""
        if method == "transformer/predict":
            return self._tool_predict(
                params.get("ehr_data", []),
                params.get("lab_tests", []),
                params.get("return_embeddings", False)
            )
        elif method == "transformer/risk_assessment":
            return self._tool_risk_assessment(
                params.get("ehr_data", []),
                params.get("lab_tests", []),
                params.get("assessment_type", "comprehensive")
            )
        elif method == "transformer/encode":
            return {
                "ehr": self._tool_encode_ehr(params.get("ehr_data", [])),
                "lab": self._tool_encode_lab_tests(params.get("lab_tests", []))
            }
        else:
            raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    server = TransformerMCPServer()
    server.run()


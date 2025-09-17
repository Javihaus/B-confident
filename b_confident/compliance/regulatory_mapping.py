"""
Regulatory Compliance Mapping for Uncertainty Quantification

Provides mapping and reporting capabilities for various regulatory frameworks
requiring uncertainty quantification and risk assessment in AI systems.

Supported Frameworks:
- EU AI Act (Article 15 - Accuracy, Robustness and Cybersecurity)
- ISO/IEC 23053:2022 (Framework for AI risk management)
- NIST AI Risk Management Framework (AI RMF 1.0)
- IEEE 2857-2021 (Privacy Engineering and Risk Assessment)
- ISO/IEC 23094-1:2023 (AI risk management)

Key Features:
- Risk categorization based on uncertainty levels
- Compliance reporting templates
- Audit trail generation
- Performance monitoring dashboards
- Regulatory requirement validation
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification for regulatory compliance"""
    MINIMAL = "minimal"
    LIMITED = "limited"
    HIGH = "high"
    UNACCEPTABLE = "unacceptable"


class ComplianceFramework(Enum):
    """Supported regulatory compliance frameworks"""
    EU_AI_ACT = "eu_ai_act"
    ISO_23053 = "iso_23053_2022"
    NIST_AI_RMF = "nist_ai_rmf_1_0"
    IEEE_2857 = "ieee_2857_2021"
    ISO_23094 = "iso_23094_1_2023"


@dataclass
class ComplianceAssessment:
    """Results of regulatory compliance assessment"""
    framework: ComplianceFramework
    risk_level: RiskLevel
    compliance_score: float  # 0-100
    requirements_met: List[str]
    requirements_failed: List[str]
    recommendations: List[str]
    assessment_timestamp: datetime
    uncertainty_metrics: Dict[str, float]


@dataclass
class AuditTrail:
    """Audit trail entry for compliance monitoring"""
    timestamp: datetime
    model_id: str
    uncertainty_score: float
    confidence_score: float
    input_hash: str
    output_hash: str
    risk_level: RiskLevel
    compliance_flags: List[str]
    metadata: Dict[str, Any]


class EUAIActCompliance:
    """
    EU AI Act Article 15 compliance mapping.

    Focuses on accuracy, robustness and cybersecurity requirements
    for high-risk AI systems.
    """

    # Risk thresholds based on Article 15 requirements
    RISK_THRESHOLDS = {
        RiskLevel.MINIMAL: 0.05,      # < 5% uncertainty
        RiskLevel.LIMITED: 0.15,      # 5-15% uncertainty
        RiskLevel.HIGH: 0.30,         # 15-30% uncertainty
        RiskLevel.UNACCEPTABLE: 1.0   # > 30% uncertainty
    }

    ARTICLE_15_REQUIREMENTS = [
        "accuracy_monitoring",
        "robustness_testing",
        "uncertainty_quantification",
        "performance_monitoring",
        "risk_assessment",
        "human_oversight",
        "transparency_reporting"
    ]

    def assess_compliance(self,
                         uncertainty_score: float,
                         accuracy_score: float,
                         calibration_score: float,
                         model_metadata: Dict[str, Any]) -> ComplianceAssessment:
        """
        Assess EU AI Act Article 15 compliance.

        Args:
            uncertainty_score: Model uncertainty [0, 1]
            accuracy_score: Model accuracy [0, 1]
            calibration_score: Calibration quality [0, 1]
            model_metadata: Additional model information

        Returns:
            Compliance assessment results
        """
        # Determine risk level based on uncertainty
        risk_level = self._classify_risk_level(uncertainty_score)

        # Check specific requirements
        requirements_met = []
        requirements_failed = []

        # Accuracy requirement (Article 15.1.a)
        if accuracy_score >= 0.85:  # 85% minimum accuracy for high-risk systems
            requirements_met.append("minimum_accuracy_threshold")
        else:
            requirements_failed.append("minimum_accuracy_threshold")

        # Uncertainty quantification (Article 15.1.b)
        if uncertainty_score <= 0.20:  # 20% max uncertainty for high-risk
            requirements_met.append("uncertainty_quantification")
        else:
            requirements_failed.append("uncertainty_quantification")

        # Calibration requirement (implicit in robustness)
        if calibration_score >= 0.90:  # Well-calibrated predictions
            requirements_met.append("prediction_calibration")
        else:
            requirements_failed.append("prediction_calibration")

        # Performance monitoring (Article 15.3)
        if model_metadata.get("monitoring_enabled", False):
            requirements_met.append("continuous_monitoring")
        else:
            requirements_failed.append("continuous_monitoring")

        # Human oversight (Article 15.2)
        if model_metadata.get("human_oversight", False):
            requirements_met.append("human_oversight")
        else:
            requirements_failed.append("human_oversight")

        # Calculate compliance score
        compliance_score = len(requirements_met) / len(self.ARTICLE_15_REQUIREMENTS) * 100

        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_level, requirements_failed, uncertainty_score
        )

        return ComplianceAssessment(
            framework=ComplianceFramework.EU_AI_ACT,
            risk_level=risk_level,
            compliance_score=compliance_score,
            requirements_met=requirements_met,
            requirements_failed=requirements_failed,
            recommendations=recommendations,
            assessment_timestamp=datetime.now(timezone.utc),
            uncertainty_metrics={
                "uncertainty_score": uncertainty_score,
                "accuracy_score": accuracy_score,
                "calibration_score": calibration_score
            }
        )

    def _classify_risk_level(self, uncertainty_score: float) -> RiskLevel:
        """Classify risk level based on uncertainty score"""
        if uncertainty_score <= self.RISK_THRESHOLDS[RiskLevel.MINIMAL]:
            return RiskLevel.MINIMAL
        elif uncertainty_score <= self.RISK_THRESHOLDS[RiskLevel.LIMITED]:
            return RiskLevel.LIMITED
        elif uncertainty_score <= self.RISK_THRESHOLDS[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        else:
            return RiskLevel.UNACCEPTABLE

    def _generate_recommendations(self,
                                risk_level: RiskLevel,
                                failed_requirements: List[str],
                                uncertainty_score: float) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []

        if risk_level == RiskLevel.UNACCEPTABLE:
            recommendations.append("IMMEDIATE ACTION REQUIRED: System exceeds acceptable uncertainty levels")
            recommendations.append("Consider model retraining or architectural changes")

        if "minimum_accuracy_threshold" in failed_requirements:
            recommendations.append("Improve model accuracy through additional training or data augmentation")

        if "uncertainty_quantification" in failed_requirements:
            recommendations.append("Implement proper uncertainty quantification (consider PBA or ensemble methods)")

        if "prediction_calibration" in failed_requirements:
            recommendations.append("Apply calibration techniques (temperature scaling, Platt scaling)")

        if "continuous_monitoring" in failed_requirements:
            recommendations.append("Implement continuous performance monitoring system")

        if "human_oversight" in failed_requirements:
            recommendations.append("Establish human oversight procedures for high-uncertainty predictions")

        return recommendations


class NISTAIRMFCompliance:
    """NIST AI Risk Management Framework compliance mapping"""

    NIST_FUNCTIONS = [
        "govern", "map", "measure", "manage"
    ]

    def assess_compliance(self,
                         uncertainty_metrics: Dict[str, float],
                         risk_management_processes: Dict[str, bool]) -> ComplianceAssessment:
        """Assess NIST AI RMF compliance"""

        requirements_met = []
        requirements_failed = []

        # GOVERN: Organization-level risk governance
        if risk_management_processes.get("governance_structure", False):
            requirements_met.append("ai_risk_governance")
        else:
            requirements_failed.append("ai_risk_governance")

        # MAP: Risk identification and categorization
        if uncertainty_metrics.get("pba_uncertainty", 0) < 0.25:
            requirements_met.append("risk_identification")
        else:
            requirements_failed.append("risk_identification")

        # MEASURE: Risk assessment and monitoring
        if uncertainty_metrics.get("calibration_score", 0) > 0.85:
            requirements_met.append("risk_measurement")
        else:
            requirements_failed.append("risk_measurement")

        # MANAGE: Risk mitigation and response
        if risk_management_processes.get("incident_response", False):
            requirements_met.append("risk_mitigation")
        else:
            requirements_failed.append("risk_mitigation")

        risk_level = RiskLevel.LIMITED  # Default for NIST framework
        compliance_score = len(requirements_met) / len(self.NIST_FUNCTIONS) * 100

        return ComplianceAssessment(
            framework=ComplianceFramework.NIST_AI_RMF,
            risk_level=risk_level,
            compliance_score=compliance_score,
            requirements_met=requirements_met,
            requirements_failed=requirements_failed,
            recommendations=["Implement comprehensive AI risk management program"],
            assessment_timestamp=datetime.now(timezone.utc),
            uncertainty_metrics=uncertainty_metrics
        )


class RegulatoryComplianceManager:
    """
    Main compliance management system supporting multiple frameworks.

    Provides unified interface for regulatory compliance assessment,
    audit trail generation, and compliance reporting.
    """

    def __init__(self):
        self.eu_ai_act = EUAIActCompliance()
        self.nist_ai_rmf = NISTAIRMFCompliance()
        self.audit_trail: List[AuditTrail] = []

    def assess_multi_framework_compliance(self,
                                        uncertainty_score: float,
                                        accuracy_score: float,
                                        calibration_score: float,
                                        model_metadata: Dict[str, Any],
                                        frameworks: List[ComplianceFramework]) -> Dict[ComplianceFramework, ComplianceAssessment]:
        """
        Assess compliance across multiple regulatory frameworks.

        Args:
            uncertainty_score: Model uncertainty [0, 1]
            accuracy_score: Model accuracy [0, 1]
            calibration_score: Calibration quality [0, 1]
            model_metadata: Model and deployment metadata
            frameworks: List of frameworks to assess

        Returns:
            Compliance assessments for each framework
        """
        results = {}

        for framework in frameworks:
            if framework == ComplianceFramework.EU_AI_ACT:
                assessment = self.eu_ai_act.assess_compliance(
                    uncertainty_score, accuracy_score, calibration_score, model_metadata
                )
                results[framework] = assessment

            elif framework == ComplianceFramework.NIST_AI_RMF:
                uncertainty_metrics = {
                    "pba_uncertainty": uncertainty_score,
                    "calibration_score": calibration_score
                }
                risk_processes = model_metadata.get("risk_management", {})
                assessment = self.nist_ai_rmf.assess_compliance(uncertainty_metrics, risk_processes)
                results[framework] = assessment

        return results

    def log_audit_trail(self,
                       model_id: str,
                       uncertainty_score: float,
                       confidence_score: float,
                       input_data: str,
                       output_data: str,
                       risk_level: RiskLevel,
                       compliance_flags: List[str],
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log entry to compliance audit trail.

        Args:
            model_id: Unique model identifier
            uncertainty_score: Model uncertainty
            confidence_score: Model confidence
            input_data: Input data (hashed for privacy)
            output_data: Output data (hashed for privacy)
            risk_level: Assessed risk level
            compliance_flags: Any compliance issues flagged
            metadata: Additional metadata
        """
        import hashlib

        # Hash sensitive data for privacy
        input_hash = hashlib.sha256(input_data.encode()).hexdigest()
        output_hash = hashlib.sha256(output_data.encode()).hexdigest()

        audit_entry = AuditTrail(
            timestamp=datetime.now(timezone.utc),
            model_id=model_id,
            uncertainty_score=uncertainty_score,
            confidence_score=confidence_score,
            input_hash=input_hash,
            output_hash=output_hash,
            risk_level=risk_level,
            compliance_flags=compliance_flags,
            metadata=metadata or {}
        )

        self.audit_trail.append(audit_entry)

    def generate_compliance_report(self,
                                 assessments: Dict[ComplianceFramework, ComplianceAssessment],
                                 model_info: Dict[str, Any]) -> str:
        """
        Generate comprehensive compliance report.

        Args:
            assessments: Compliance assessments for each framework
            model_info: Model metadata and information

        Returns:
            Formatted compliance report
        """
        report_lines = [
            "=== REGULATORY COMPLIANCE REPORT ===",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Model ID: {model_info.get('model_id', 'Unknown')}",
            f"Model Version: {model_info.get('version', 'Unknown')}",
            ""
        ]

        # Summary section
        overall_compliance = np.mean([a.compliance_score for a in assessments.values()])
        max_risk = max([a.risk_level for a in assessments.values()], key=lambda x: list(RiskLevel).index(x))

        report_lines.extend([
            "EXECUTIVE SUMMARY:",
            f"  Overall Compliance Score: {overall_compliance:.1f}%",
            f"  Maximum Risk Level: {max_risk.value.upper()}",
            f"  Frameworks Assessed: {len(assessments)}",
            ""
        ])

        # Framework-specific assessments
        for framework, assessment in assessments.items():
            report_lines.extend([
                f"{framework.value.upper()} ASSESSMENT:",
                f"  Compliance Score: {assessment.compliance_score:.1f}%",
                f"  Risk Level: {assessment.risk_level.value.upper()}",
                f"  Requirements Met: {len(assessment.requirements_met)}",
                f"  Requirements Failed: {len(assessment.requirements_failed)}",
                ""
            ])

            if assessment.requirements_failed:
                report_lines.extend([
                    "  Failed Requirements:",
                    *[f"    • {req}" for req in assessment.requirements_failed],
                    ""
                ])

            if assessment.recommendations:
                report_lines.extend([
                    "  Recommendations:",
                    *[f"    • {rec}" for rec in assessment.recommendations],
                    ""
                ])

        # Uncertainty metrics summary
        if assessments:
            sample_assessment = next(iter(assessments.values()))
            report_lines.extend([
                "UNCERTAINTY METRICS:",
                f"  PBA Uncertainty: {sample_assessment.uncertainty_metrics.get('uncertainty_score', 'N/A')}",
                f"  Model Accuracy: {sample_assessment.uncertainty_metrics.get('accuracy_score', 'N/A')}",
                f"  Calibration Score: {sample_assessment.uncertainty_metrics.get('calibration_score', 'N/A')}",
                ""
            ])

        # Action items
        all_recommendations = []
        for assessment in assessments.values():
            all_recommendations.extend(assessment.recommendations)

        if all_recommendations:
            report_lines.extend([
                "PRIORITY ACTION ITEMS:",
                *[f"  {i+1}. {rec}" for i, rec in enumerate(set(all_recommendations))],
                ""
            ])

        report_lines.extend([
            "COMPLIANCE STATUS:",
            f"  ✓ COMPLIANT" if overall_compliance >= 90 else f"  ⚠ REQUIRES ATTENTION" if overall_compliance >= 70 else f"  ✗ NON-COMPLIANT",
            ""
        ])

        return "\n".join(report_lines)

    def export_audit_trail(self, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Export audit trail for compliance reporting.

        Args:
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            List of audit trail entries as dictionaries
        """
        filtered_trail = self.audit_trail

        if start_date:
            filtered_trail = [entry for entry in filtered_trail if entry.timestamp >= start_date]
        if end_date:
            filtered_trail = [entry for entry in filtered_trail if entry.timestamp <= end_date]

        return [
            {
                "timestamp": entry.timestamp.isoformat(),
                "model_id": entry.model_id,
                "uncertainty_score": entry.uncertainty_score,
                "confidence_score": entry.confidence_score,
                "input_hash": entry.input_hash,
                "output_hash": entry.output_hash,
                "risk_level": entry.risk_level.value,
                "compliance_flags": entry.compliance_flags,
                "metadata": entry.metadata
            }
            for entry in filtered_trail
        ]

    def get_compliance_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for compliance monitoring dashboard.

        Returns:
            Dashboard data including trends and alerts
        """
        if not self.audit_trail:
            return {"status": "No data available"}

        recent_entries = sorted(self.audit_trail, key=lambda x: x.timestamp, reverse=True)[:100]

        # Calculate trends
        avg_uncertainty = np.mean([entry.uncertainty_score for entry in recent_entries])
        avg_confidence = np.mean([entry.confidence_score for entry in recent_entries])

        # Risk level distribution
        risk_distribution = {}
        for level in RiskLevel:
            count = sum(1 for entry in recent_entries if entry.risk_level == level)
            risk_distribution[level.value] = count

        # Compliance flags frequency
        flag_frequency = {}
        for entry in recent_entries:
            for flag in entry.compliance_flags:
                flag_frequency[flag] = flag_frequency.get(flag, 0) + 1

        return {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_assessments": len(self.audit_trail),
            "recent_assessments": len(recent_entries),
            "average_uncertainty": avg_uncertainty,
            "average_confidence": avg_confidence,
            "risk_distribution": risk_distribution,
            "common_compliance_flags": dict(sorted(flag_frequency.items(),
                                                  key=lambda x: x[1], reverse=True)[:10]),
            "alert_status": "HIGH_RISK" if avg_uncertainty > 0.3 else "NORMAL"
        }
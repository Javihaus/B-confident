#!/usr/bin/env python3
"""
Operational Decision-Making Value Demonstrations
Concrete examples of how uncertainty scores guide production decisions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"

@dataclass
class OperationalDecision:
    action: str
    reasoning: str
    cost_impact: str
    risk_level: str
    human_involvement: str

class OperationalDecisionMaker:
    """Demonstrates operational decision-making based on uncertainty scores"""

    def __init__(self):
        self.confidence_thresholds = {
            ConfidenceLevel.HIGH: 0.3,      # < 0.3 uncertainty
            ConfidenceLevel.MEDIUM: 0.6,    # 0.3-0.6 uncertainty
            ConfidenceLevel.LOW: 0.8,       # 0.6-0.8 uncertainty
            ConfidenceLevel.CRITICAL: 1.0   # > 0.8 uncertainty
        }

    def classify_confidence(self, uncertainty_score: float) -> ConfidenceLevel:
        """Classify confidence level based on uncertainty score"""
        if uncertainty_score < self.confidence_thresholds[ConfidenceLevel.HIGH]:
            return ConfidenceLevel.HIGH
        elif uncertainty_score < self.confidence_thresholds[ConfidenceLevel.MEDIUM]:
            return ConfidenceLevel.MEDIUM
        elif uncertainty_score < self.confidence_thresholds[ConfidenceLevel.LOW]:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.CRITICAL

    def make_content_moderation_decision(self, text: str, uncertainty: float) -> OperationalDecision:
        """Content moderation decision based on uncertainty"""
        confidence = self.classify_confidence(uncertainty)

        if confidence == ConfidenceLevel.HIGH:
            return OperationalDecision(
                action="Auto-approve content",
                reasoning=f"High confidence (uncertainty={uncertainty:.3f}) in classification",
                cost_impact="Minimal - automated processing",
                risk_level="Low - reliable prediction",
                human_involvement="None required"
            )
        elif confidence == ConfidenceLevel.MEDIUM:
            return OperationalDecision(
                action="Flag for expedited review",
                reasoning=f"Medium confidence (uncertainty={uncertainty:.3f}) requires validation",
                cost_impact="Low - quick human check",
                risk_level="Medium - potential edge case",
                human_involvement="Brief expert review (2-3 minutes)"
            )
        elif confidence == ConfidenceLevel.LOW:
            return OperationalDecision(
                action="Queue for detailed review",
                reasoning=f"Low confidence (uncertainty={uncertainty:.3f}) indicates complex case",
                cost_impact="Medium - thorough human analysis",
                risk_level="High - uncertain classification",
                human_involvement="Full expert analysis (10-15 minutes)"
            )
        else:
            return OperationalDecision(
                action="Escalate to senior moderator",
                reasoning=f"Critical uncertainty (uncertainty={uncertainty:.3f}) suggests novel content",
                cost_impact="High - senior expert time",
                risk_level="Critical - potential policy gap",
                human_involvement="Senior moderator + policy team review"
            )

    def make_customer_service_decision(self, query: str, uncertainty: float) -> OperationalDecision:
        """Customer service routing decision based on uncertainty"""
        confidence = self.classify_confidence(uncertainty)

        if confidence == ConfidenceLevel.HIGH:
            return OperationalDecision(
                action="Provide automated response",
                reasoning=f"High confidence (uncertainty={uncertainty:.3f}) in answer accuracy",
                cost_impact="$0.01 per query - automated",
                risk_level="Low - reliable answer",
                human_involvement="None - automated response"
            )
        elif confidence == ConfidenceLevel.MEDIUM:
            return OperationalDecision(
                action="Automated response + satisfaction check",
                reasoning=f"Medium confidence (uncertainty={uncertainty:.3f}) needs validation",
                cost_impact="$0.05 per query - automated + follow-up",
                risk_level="Medium - monitor satisfaction",
                human_involvement="Automated satisfaction survey"
            )
        elif confidence == ConfidenceLevel.LOW:
            return OperationalDecision(
                action="Route to human agent",
                reasoning=f"Low confidence (uncertainty={uncertainty:.3f}) requires human expertise",
                cost_impact="$2.50 per query - human agent",
                risk_level="High - complex query",
                human_involvement="Trained customer service agent"
            )
        else:
            return OperationalDecision(
                action="Escalate to specialist",
                reasoning=f"Critical uncertainty (uncertainty={uncertainty:.3f}) indicates complex issue",
                cost_impact="$8.00 per query - specialist time",
                risk_level="Critical - potential escalation",
                human_involvement="Domain specialist + possible supervisor"
            )

    def make_medical_advice_decision(self, query: str, uncertainty: float) -> OperationalDecision:
        """Medical advice routing decision (high-stakes scenario)"""
        confidence = self.classify_confidence(uncertainty)

        if confidence == ConfidenceLevel.HIGH:
            return OperationalDecision(
                action="Provide information with disclaimer",
                reasoning=f"High confidence (uncertainty={uncertainty:.3f}) in informational content",
                cost_impact="Minimal - automated with legal disclaimers",
                risk_level="Low - general information only",
                human_involvement="None - automated with clear disclaimers"
            )
        elif confidence == ConfidenceLevel.MEDIUM:
            return OperationalDecision(
                action="Provide information + professional consultation advice",
                reasoning=f"Medium confidence (uncertainty={uncertainty:.3f}) requires professional guidance",
                cost_impact="Low - automated + referral system",
                risk_level="Medium - recommend professional consultation",
                human_involvement="Automated referral to healthcare provider"
            )
        elif confidence == ConfidenceLevel.LOW:
            return OperationalDecision(
                action="Direct to healthcare professional immediately",
                reasoning=f"Low confidence (uncertainty={uncertainty:.3f}) in medical domain",
                cost_impact="None - immediate referral",
                risk_level="High - potential medical complexity",
                human_involvement="Immediate healthcare provider consultation"
            )
        else:
            return OperationalDecision(
                action="Emergency protocol activation",
                reasoning=f"Critical uncertainty (uncertainty={uncertainty:.3f}) suggests emergency situation",
                cost_impact="High - emergency response system",
                risk_level="Critical - potential emergency",
                human_involvement="Emergency services + immediate medical attention"
            )

    def make_financial_advice_decision(self, query: str, uncertainty: float) -> OperationalDecision:
        """Financial advice routing decision (regulatory compliance)"""
        confidence = self.classify_confidence(uncertainty)

        if confidence == ConfidenceLevel.HIGH:
            return OperationalDecision(
                action="Provide educational information",
                reasoning=f"High confidence (uncertainty={uncertainty:.3f}) in educational content",
                cost_impact="Minimal - automated educational resources",
                risk_level="Low - general financial education",
                human_involvement="None - educational content delivery"
            )
        elif confidence == ConfidenceLevel.MEDIUM:
            return OperationalDecision(
                action="Educational content + advisor referral",
                reasoning=f"Medium confidence (uncertainty={uncertainty:.3f}) suggests need for personalization",
                cost_impact="Low - automated + referral system",
                risk_level="Medium - potential personalized advice needed",
                human_involvement="Qualified financial advisor referral"
            )
        elif confidence == ConfidenceLevel.LOW:
            return OperationalDecision(
                action="Direct to certified financial planner",
                reasoning=f"Low confidence (uncertainty={uncertainty:.3f}) requires professional expertise",
                cost_impact="None - immediate referral",
                risk_level="High - complex financial situation",
                human_involvement="Certified Financial Planner consultation"
            )
        else:
            return OperationalDecision(
                action="Escalate to compliance team",
                reasoning=f"Critical uncertainty (uncertainty={uncertainty:.3f}) may involve regulatory issues",
                cost_impact="High - compliance review",
                risk_level="Critical - potential regulatory complexity",
                human_involvement="Compliance team + legal review if needed"
            )

    def calculate_roi_analysis(self, decisions: List[OperationalDecision], volume: int) -> Dict:
        """Calculate ROI analysis for operational decisions"""

        # Cost estimates per decision type
        cost_mapping = {
            "Auto-approve content": 0.01,
            "Flag for expedited review": 0.25,
            "Queue for detailed review": 1.50,
            "Escalate to senior moderator": 5.00,
            "Provide automated response": 0.01,
            "Automated response + satisfaction check": 0.05,
            "Route to human agent": 2.50,
            "Escalate to specialist": 8.00,
            "Provide information with disclaimer": 0.01,
            "Provide information + professional consultation advice": 0.10,
            "Direct to healthcare professional immediately": 0.00,
            "Emergency protocol activation": 25.00,
            "Provide educational information": 0.01,
            "Educational content + advisor referral": 0.15,
            "Direct to certified financial planner": 0.00,
            "Escalate to compliance team": 12.00
        }

        total_cost = 0
        decision_breakdown = {}

        for decision in decisions:
            cost_per_unit = cost_mapping.get(decision.action, 1.00)
            total_cost += cost_per_unit

            if decision.action in decision_breakdown:
                decision_breakdown[decision.action] += 1
            else:
                decision_breakdown[decision.action] = 1

        # Compare to baseline (all human review)
        baseline_cost = volume * 3.00  # Average human review cost
        uncertainty_guided_cost = total_cost * volume

        savings = baseline_cost - uncertainty_guided_cost
        roi_percentage = (savings / uncertainty_guided_cost) * 100 if uncertainty_guided_cost > 0 else 0

        return {
            "total_volume": volume,
            "uncertainty_guided_cost": uncertainty_guided_cost,
            "baseline_cost": baseline_cost,
            "cost_savings": savings,
            "roi_percentage": roi_percentage,
            "decision_breakdown": decision_breakdown,
            "cost_per_decision": uncertainty_guided_cost / volume if volume > 0 else 0
        }

def demonstrate_operational_value():
    """Demonstrate operational decision-making value across domains"""

    decision_maker = OperationalDecisionMaker()

    # Simulate realistic uncertainty distributions for different scenarios
    scenarios = {
        "Content Moderation": {
            "examples": [
                ("Standard news article", 0.15),
                ("User comment with slang", 0.35),
                ("Borderline harassment case", 0.65),
                ("Novel hate speech variant", 0.85)
            ],
            "decision_func": decision_maker.make_content_moderation_decision
        },
        "Customer Service": {
            "examples": [
                ("Password reset request", 0.12),
                ("Billing inquiry", 0.28),
                ("Technical support issue", 0.58),
                ("Complex policy interpretation", 0.82)
            ],
            "decision_func": decision_maker.make_customer_service_decision
        },
        "Medical Information": {
            "examples": [
                ("General health information", 0.20),
                ("Medication side effects", 0.45),
                ("Symptom interpretation", 0.70),
                ("Emergency medical situation", 0.90)
            ],
            "decision_func": decision_maker.make_medical_advice_decision
        },
        "Financial Advisory": {
            "examples": [
                ("Basic budgeting advice", 0.18),
                ("Investment terminology", 0.40),
                ("Complex tax situation", 0.68),
                ("Regulatory compliance issue", 0.88)
            ],
            "decision_func": decision_maker.make_financial_advice_decision
        }
    }

    results = {}

    for domain, config in scenarios.items():
        print(f"\n{'='*60}")
        print(f"OPERATIONAL DECISIONS: {domain.upper()}")
        print(f"{'='*60}")

        domain_decisions = []

        for example_text, uncertainty in config["examples"]:
            decision = config["decision_func"](example_text, uncertainty)
            domain_decisions.append(decision)

            print(f"\nScenario: {example_text}")
            print(f"Uncertainty Score: {uncertainty:.3f}")
            print(f"→ Decision: {decision.action}")
            print(f"  Reasoning: {decision.reasoning}")
            print(f"  Cost Impact: {decision.cost_impact}")
            print(f"  Risk Level: {decision.risk_level}")
            print(f"  Human Involvement: {decision.human_involvement}")

        # ROI Analysis for this domain
        roi_analysis = decision_maker.calculate_roi_analysis(domain_decisions, 10000)
        results[domain] = roi_analysis

        print(f"\n{'-'*40}")
        print(f"ROI ANALYSIS ({domain})")
        print(f"{'-'*40}")
        print(f"Volume: {roi_analysis['total_volume']:,} queries/month")
        print(f"Uncertainty-Guided Cost: ${roi_analysis['uncertainty_guided_cost']:,.2f}")
        print(f"Baseline (All Human) Cost: ${roi_analysis['baseline_cost']:,.2f}")
        print(f"Monthly Savings: ${roi_analysis['cost_savings']:,.2f}")
        print(f"ROI: {roi_analysis['roi_percentage']:.1f}%")
        print(f"Cost per Decision: ${roi_analysis['cost_per_decision']:.3f}")

    # Overall summary
    print(f"\n{'='*60}")
    print("SUMMARY: OPERATIONAL VALUE OF UNCERTAINTY QUANTIFICATION")
    print(f"{'='*60}")

    total_savings = sum(r['cost_savings'] for r in results.values())
    total_volume = sum(r['total_volume'] for r in results.values())

    print(f"\nCross-Domain Impact:")
    print(f"• Total Monthly Volume: {total_volume:,} decisions")
    print(f"• Total Monthly Savings: ${total_savings:,.2f}")
    print(f"• Annual Savings: ${total_savings * 12:,.2f}")

    print(f"\nKey Operational Benefits:")
    print(f"• Automated Processing: High-confidence decisions (uncertainty < 0.3)")
    print(f"• Cost Optimization: Route complex cases to appropriate expertise level")
    print(f"• Risk Management: Prevent high-risk automated decisions")
    print(f"• Regulatory Compliance: Systematic uncertainty documentation")
    print(f"• Quality Assurance: Human oversight focused on uncertain cases")

    print(f"\nProduction Infrastructure Value:")
    print(f"• Essential for EU AI Act Article 15 compliance")
    print(f"• Enables reliable deployment of current transformer architectures")
    print(f"• Creates measurement infrastructure for future uncertainty-aware models")
    print(f"• Provides operational decision-making protocols for regulated environments")

    return results

if __name__ == "__main__":
    demonstrate_operational_value()
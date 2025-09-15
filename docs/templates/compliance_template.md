# EU AI Act Article 15 Compliance Template

This template provides guidance for organizations implementing uncertainty quantification
to meet EU AI Act Article 15 requirements.

## Article 15 Requirements Summary

The EU AI Act Article 15 requires high-risk AI systems to:

1. **Achieve appropriate level of accuracy** relative to intended purpose
2. **Implement relevant accuracy metrics** for system evaluation
3. **Maintain consistent performance throughout the system lifecycle**

## Compliance Implementation with PBA

### 1. Appropriate Level of Accuracy

**Requirement**: Systems must achieve accuracy levels appropriate to their intended purpose.

**PBA Implementation**:
```python
from uncertainty_pba import calibrate_model, PBAConfig

# Use paper-validated optimal parameters
config = PBAConfig(alpha=0.9, beta=0.5)

# Calibrate on domain-specific validation data
results = calibrate_model(
    model="your-model",
    validation_texts=domain_validation_texts,
    validation_labels=correctness_labels,
    pba_config=config
)

# Document calibration quality
ece = results['calibration_results'].ece
if ece < 0.03:
    calibration_quality = "Excellent (ECE < 3%)"
elif ece < 0.05:
    calibration_quality = "Good (ECE < 5%)"
else:
    calibration_quality = "Requires improvement"
```

**Documentation Requirements**:
- Expected Calibration Error measurement
- Validation dataset description
- Domain-specific accuracy thresholds
- Calibration quality assessment

### 2. Relevant Accuracy Metrics

**Requirement**: Implement appropriate metrics for measuring system accuracy.

**PBA Implementation**:
```python
from uncertainty_pba import uncertainty_metrics

# Comprehensive metrics calculation
metrics = uncertainty_metrics(
    uncertainty_scores=pba_uncertainties,
    correctness_labels=ground_truth_labels,
    include_cross_validation=True
)

# Required metrics for compliance
required_metrics = {
    'expected_calibration_error': metrics['ece'],
    'brier_score': metrics['brier_score'],
    'auroc': metrics['auroc'],
    'stability_score': metrics['stability_score']
}
```

**Documentation Requirements**:
- Expected Calibration Error (ECE)
- Brier Score for probabilistic accuracy
- Area Under ROC Curve (AUROC) for discrimination
- Stability Score for cross-validation consistency

### 3. Consistent Performance Throughout Lifecycle

**Requirement**: Maintain performance consistency from deployment through operation.

**PBA Implementation**:
```python
from uncertainty_pba import create_continuous_monitor

# Set up production monitoring
monitor = create_continuous_monitor(
    baseline_results=initial_calibration_results,
    alert_thresholds={
        'ece_warning': 1.5,    # 50% increase triggers warning
        'ece_critical': 2.0,   # 100% increase triggers critical alert
        'brier_warning': 1.3,
        'auroc_warning': 0.9
    }
)

# Continuous monitoring in production
def production_inference_loop():
    while True:
        # Process batch of requests
        uncertainties, labels = process_batch()

        # Monitor for drift
        alerts = monitor.add_samples(uncertainties, labels)

        # Handle alerts
        for alert in alerts:
            if alert.alert_level == "CRITICAL":
                trigger_recalibration()
                notify_compliance_team()
```

**Documentation Requirements**:
- Baseline performance metrics
- Monitoring procedures and thresholds
- Drift detection methodology
- Recalibration procedures

## Compliance Report Generation

### Automated Report Generation
```python
from uncertainty_pba import compliance_report

# Generate comprehensive compliance report
report = compliance_report(
    system_name="YourAISystem",
    calibration_results=calibration_data,
    system_version="1.0",
    evaluation_dataset="production_validation_v1",
    model_architecture="Architecture description",
    output_format="markdown",
    save_path="compliance_reports/eu_ai_act_report.md"
)
```

### Manual Documentation Checklist

#### System Description
- [ ] AI system name and version
- [ ] Intended purpose and use cases
- [ ] Model architecture and training details
- [ ] Deployment environment description

#### Accuracy Declaration
- [ ] PBA methodology implementation
- [ ] Calibration quality assessment
- [ ] Domain-specific accuracy requirements
- [ ] Validation procedures description

#### Metrics Documentation
- [ ] Expected Calibration Error measurement
- [ ] Brier Score calculation
- [ ] AUROC discrimination ability
- [ ] Cross-validation stability analysis
- [ ] Statistical significance testing

#### Lifecycle Consistency
- [ ] Baseline performance establishment
- [ ] Continuous monitoring procedures
- [ ] Drift detection thresholds
- [ ] Recalibration protocols
- [ ] Alert escalation procedures

## Risk Assessment Template

### High-Risk Scenarios
Identify scenarios where uncertainty quantification is critical:

1. **Medical Diagnosis Support**
   - Required ECE: < 0.02 (2%)
   - Monitoring frequency: Real-time
   - Recalibration trigger: 25% ECE increase

2. **Financial Decision Support**
   - Required ECE: < 0.03 (3%)
   - Monitoring frequency: Daily
   - Recalibration trigger: 50% ECE increase

3. **Legal Document Analysis**
   - Required ECE: < 0.05 (5%)
   - Monitoring frequency: Weekly
   - Recalibration trigger: 75% ECE increase

### Risk Mitigation Strategies

```python
def implement_risk_mitigation():
    """Implementation template for risk mitigation"""

    # 1. Establish baseline performance
    baseline_calibration = calibrate_model(
        model, validation_data, validation_labels
    )

    # 2. Set appropriate thresholds
    if use_case == "medical":
        ece_threshold = 0.02
        alert_multiplier = 1.25  # 25% increase
    elif use_case == "financial":
        ece_threshold = 0.03
        alert_multiplier = 1.50  # 50% increase
    else:
        ece_threshold = 0.05
        alert_multiplier = 1.75  # 75% increase

    # 3. Implement monitoring
    monitor = create_continuous_monitor(
        baseline_calibration,
        alert_thresholds={'ece_critical': alert_multiplier}
    )

    # 4. Define response procedures
    def handle_performance_degradation():
        # Immediate: Increase human oversight
        # Short-term: Recalibrate on recent data
        # Long-term: Retrain or replace model
        pass
```

## Audit Trail Requirements

### Documentation to Maintain
1. **Model Versioning**
   - Model checkpoints and versions
   - Training data descriptions
   - Hyperparameter configurations
   - PBA configuration parameters

2. **Validation Records**
   - Validation dataset descriptions
   - Calibration analysis results
   - Cross-validation outcomes
   - Statistical significance tests

3. **Production Monitoring**
   - Continuous monitoring logs
   - Alert history and responses
   - Recalibration records
   - Performance trend analysis

4. **Compliance Documentation**
   - Generated compliance reports
   - Regulatory correspondence
   - Audit findings and responses
   - Process improvement records

### Retention Periods
- Calibration results: 7 years minimum
- Monitoring logs: 5 years minimum
- Compliance reports: 10 years minimum
- Audit documentation: Permanent

## Integration with Existing QMS

### Quality Management System Integration
```python
class ComplianceQMS:
    """Integration with Quality Management System"""

    def __init__(self):
        self.compliance_checker = ComplianceChecker()
        self.audit_logger = AuditLogger()

    def validate_deployment(self, model, validation_data):
        """Pre-deployment compliance validation"""

        # Run calibration analysis
        results = calibrate_model(model, validation_data)

        # Check compliance thresholds
        compliance_status = self.compliance_checker.validate(results)

        # Log for audit trail
        self.audit_logger.log_validation(model, results, compliance_status)

        return compliance_status.approved

    def monitor_production(self, model_id):
        """Ongoing production compliance monitoring"""

        monitor = self.get_monitor(model_id)
        alerts = monitor.check_recent_performance()

        for alert in alerts:
            self.audit_logger.log_alert(model_id, alert)
            self.notify_stakeholders(model_id, alert)
```

## Stakeholder Communication

### Technical Team Communication
- Calibration metrics and trends
- Alert notifications and responses
- Model performance analysis
- Recalibration recommendations

### Compliance Team Communication
- Regulatory status updates
- Compliance report summaries
- Risk assessment updates
- Audit preparation materials

### Executive Communication
- High-level compliance status
- Business impact of performance changes
- Resource requirements for compliance
- Strategic recommendations

## Continuous Improvement

### Performance Review Cycle
1. **Monthly**: Review monitoring alerts and trends
2. **Quarterly**: Comprehensive calibration reassessment
3. **Annually**: Full compliance audit and documentation review
4. **Ad-hoc**: Response to regulatory updates or significant alerts

### Process Enhancement
- Regular review of calibration thresholds
- Optimization of monitoring procedures
- Enhancement of documentation processes
- Training updates for staff

---

*This template should be customized for your specific use case, risk profile, and regulatory requirements. Consult with legal and compliance experts for implementation guidance.*
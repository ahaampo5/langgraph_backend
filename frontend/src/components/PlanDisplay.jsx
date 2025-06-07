import React from 'react';
import './PlanDisplay.css';

const PlanDisplay = ({ plan }) => {
  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return '✅';
      case 'in_progress':
        return '⏳';
      case 'pending':
        return '📋';
      default:
        return '⚪';
    }
  };

  const getStatusClass = (status) => {
    switch (status) {
      case 'completed':
        return 'completed';
      case 'in_progress':
        return 'in-progress';
      case 'pending':
        return 'pending';
      default:
        return 'unknown';
    }
  };

  return (
    <div className="plan-display">
      <div className="plan-header">
        <h4 className="plan-title">📋 실행 계획</h4>
        <div className="plan-goal">
          <strong>목표:</strong> {plan.goal}
        </div>
      </div>
      
      <div className="plan-steps">
        <h5 className="steps-title">단계별 진행상황:</h5>
        {plan.steps.map((step) => (
          <div key={step.step_id} className={`plan-step ${getStatusClass(step.status)}`}>
            <div className="step-header">
              <span className="step-status">
                {getStatusIcon(step.status)}
              </span>
              <span className="step-description">
                <strong>단계 {step.step_id}:</strong> {step.description}
              </span>
            </div>
            
            {step.result && (
              <div className="step-result">
                <strong>결과:</strong> {step.result}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default PlanDisplay;

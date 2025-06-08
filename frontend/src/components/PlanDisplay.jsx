import React from 'react';

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
    const baseClass = "p-3 rounded-lg border-l-4 transition-all duration-300";
    switch (status) {
      case 'completed':
        return `${baseClass} bg-green-50 border-l-green-500`;
      case 'in_progress':
        return `${baseClass} bg-yellow-50 border-l-yellow-500 animate-pulse-slow`;
      case 'pending':
        return `${baseClass} bg-gray-50 border-l-gray-400`;
      default:
        return `${baseClass} bg-gray-25 border-l-gray-300`;
    }
  };

  return (
    <div className="w-full">
      <div className="mb-4">
        <h4 className="m-0 mb-2 text-purple-700 text-lg">📋 실행 계획</h4>
        <div className="p-2 bg-purple-100 rounded-lg text-sm">
          <strong>목표:</strong> {plan.goal}
        </div>
      </div>
      
      <div className="flex flex-col gap-2">
        <h5 className="m-0 mb-3 text-purple-700 text-base">단계별 진행상황:</h5>
        {plan.steps.map((step) => (
          <div key={step.step_id} className={getStatusClass(step.status)}>
            <div className="flex items-start gap-2">
              <span className="text-lg min-w-6">
                {getStatusIcon(step.status)}
              </span>
              <span className="flex-1 leading-normal">
                <strong>단계 {step.step_id}:</strong> {step.description}
              </span>
            </div>
            
            {step.result && (
              <div className="mt-2 p-2 bg-black/5 rounded text-sm leading-snug">
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

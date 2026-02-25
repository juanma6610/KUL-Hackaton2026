const API_BASE_URL = 'http://localhost:5000';

const DEFAULT_PREDICTIONS = {
  xgboost: 2.0,
  pimsleur: 2.0,
  leitner: 2.0,
  hlr: 2.0,
  shap_values: []
};

async function apiRequest(endpoint, options = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options
    });
    return await response.json();
  } catch (err) {
    console.error(`API error (${endpoint}):`, err);
    return null;
  }
}

export async function predictHalfLife(features) {
  const data = await apiRequest('/predict', {
    method: 'POST',
    body: JSON.stringify(features)
  });
  
  return data ? {
    xgboost: data.h_xgboost,
    pimsleur: data.h_pimsleur,
    leitner: data.h_leitner,
    hlr: data.h_hlr,
    shap_values: data.shap_values
  } : DEFAULT_PREDICTIONS;
}

export async function fetchUserStats() {
  return await apiRequest('/user_stats');
}

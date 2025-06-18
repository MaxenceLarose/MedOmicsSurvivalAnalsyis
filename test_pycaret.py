# Functional
from pycaret.datasets import get_data
from pycaret.regression import setup, create_model, tune_model, predict_model

# OOP
from pycaret.regression import RegressionExperiment


boston = get_data('boston')
exp_name = setup(data=boston, target='medv')
lr = create_model('lr')
tuned_lr = tune_model(lr)
pred = predict_model(lr)
print(pred)



# Functional or OOP?

### Functional ###
# Create 'setup', 'create_model', 'tune_model', 'plot_model'
# But much more... 'compare_models', 'deploy model', 'save_model', etc...

### OOP ###
# Inherit from 'RegressionExperiment' to create 'SurvivalAnalysisExperiment'


# It depends on what you actually need in MedOmics or more like what you let the user use.
# It might be better to not integrate it with pycaret but with MedOmics only?
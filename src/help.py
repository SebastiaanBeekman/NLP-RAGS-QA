from sklearn.metrics import r2_score
import numpy as np
from library import plot_ascii, setup_gp
from library import PlusNode, MinusNode, TimesNode, DivNode
from library import FeatureNode, ConstantNode
import numpy as np


### SET UP: DO NOT CHANGE ANYTHING HERE ###

# set up the seed 
SEED = 42
np.random.seed(42)

# ground truth equation linking input to output (do not change!)
def ground_truth_equation(X, measurement_noise=0.1):
    y = np.cos(X).reshape(-1)
    # let's also add some noise to simulate real-world measurement
    y += np.random.normal(scale=measurement_noise, size=len(X))
    return y

# training set (do not change!)
X_train = np.linspace(-4,4,100).reshape((-1,1))
y_train = ground_truth_equation(X_train)

# validation set (points ordered for plotting later on) (do not change!)
X_val = np.random.uniform(size=100)*8 - 4
X_val = np.sort(X_val).reshape((-1,1))
y_val = ground_truth_equation(X_val)

# set up terminal set (do not change!)
terminal_set = [FeatureNode(0), ConstantNode()]

#TODO: Set up function set
# Add the MinusNode, TimesNode and DivNode
# (note do not change the order!)
function_set = [PlusNode()]

def linear_scaling(y, prediction):
    # Your linear scaling implementation here:
    intercept = 0. # TODO: change intercept calculation
    slope = 1. # TODO: change slope calculation
    return intercept, slope

gp = setup_gp(linear_scaling, function_set, terminal_set, pop_size=40) # pop size should be multiple of 8 here due to tournaments (do not change!)

### END SETUP ###

### RUN EVOLUTION ###
# TODO: use "gp.evolve", passing the correct split of the data set
gp.evolve(X_train, y_train)

### CHECK SOLUTION QUALITY ###
# TODO: call gp.predict(...), passing the correct split of the data set 
# (note: gp.predict internally refers the call to the best-found solution)
prediction = gp.predict(X_val)

prediction_mean = np.mean(prediction)
y_val_mean = np.mean(y_val)
b = np.sum((y_val - y_val_mean) * (prediction - prediction_mean)) / np.sum((prediction - prediction_mean)**2)
b = 1 if np.isnan(b) or np.isinf(b) else b
a = y_val_mean - b * prediction_mean
prediction = a + b * prediction

# TODO: Show the R2 Score using the meaningful split (documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
r2 = r2_score(y_val, prediction) # Note: do NOT delete the variable "r2", it is used in the automatic tests
print("\nQuality (R2 Score): {:.3f}".format(r2))


### PLOTTING (OPTIONAL) ###
# you can plot how the true function and the evolved one look like, up to you what split to plot 
# (note: depending on your screen size, you might want to change width and height)
print("\nTrue function")
plot_ascii(X_train,y_train, w=100, h=20)
print("\nEvolved function")
plot_ascii(X_val,prediction, w=100, h=20)
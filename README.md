# Adaptiv_hazi

Todo:

- create abstract class "Layer"
- fix gradient descent "trainable layer" check
- get rid of ReLu layer -> implement inside conv layer (optional)
- make dense layer have optional activation (if None, replace it with *1)
- back up and refactor

- merge one_class with model -> pick cost function?
- one class should use (n,1) dimention instaed of (n,) to get rid of [:,0] every time on predict
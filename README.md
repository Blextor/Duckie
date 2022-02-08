# DuckieProject - Team Adja
Kovács Boldizsár GVJY8E

Schneider Marcell DBGYVI

Talpos Norbert Q2H4XB

Virág József Ádám U7KC0P

DEMO: https://youtu.be/8hvmge7oYk4

## Milestone 1:

Links:

For the setup : https://docs.duckietown.org/daffy/AIDO/draft/index.html

GitHub for Training and Data: https://github.com/duckietown/gym-duckietown
                       , and: https://github.com/duckietown/challenge-aido_LF-baseline-behavior-cloning

Data collection is done in the commit method of log_util.py, from where the data of the current step is accessed via a Step data structure. The observation associated with the step is saved as an image, the associated action is inserted in the my_app.txt file, all tagged for clarity.
Once the data is collected, we use detector.py to transform the images into a format more suitable for teaching, filtering out the information of interest (bisector, edge of the road). This is achieved by hsv filtering and by clipping the horizon.
For teaching, we convert the images back to a numpy array format with shape of (window_width, window_height, 3). These images become the input (X) database. The labels (y) are the lines of my_app.txt, these are the actions retrieved from the simulator. From these, we have generated the teaching, validation and test databases for training.



  automatic.py: [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="100"/>](https://colab.research.google.com/drive/17ZmFWd9ipcPhu3UMql5EZ32AyhlOysG2)

  detector.py : [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="100"/>](https://colab.research.google.com/drive/1xQSpIAknsp-DMxFXMpcI60WFaWCoqjEi)
  
  log_util.py : [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="100"/>](https://colab.research.google.com/drive/1kUI_Ohr98yPwcjAsObPhGv1oSjUGq4mM)

  data_process: [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="100"/>](https://colab.research.google.com/drive/1O8lRYQlKN9IQgttoQGnu35wppqE9DZBH)
  
  Data 1 : [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Google_Drive_icon_%282020%29.svg/1147px-Google_Drive_icon_%282020%29.svg.png" width="20"/>](https://drive.google.com/drive/folders/124WPRwzaz-ePeScy4qqRwlmeeOi_Ii7w?usp=sharing)
  
  Data 2 : [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Google_Drive_icon_%282020%29.svg/1147px-Google_Drive_icon_%282020%29.svg.png" width="20"/>](https://drive.google.com/file/d/1-Hm0SgFPcqoTUjNcBEBRo7vR-5io4Rxk/view?usp=sharing)
  
  Data 3 : [<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Google_Drive_icon_%282020%29.svg/1147px-Google_Drive_icon_%282020%29.svg.png" width="20"/>](https://drive.google.com/file/d/1agHd80lq5hrRZEONezmFWVvw3y8VbPF1/view?usp=sharing)

## Milestone 2:

Having collected a certain amount of observations from the duckietown environment, we started creating our model which learns using Imitation learning. Imitation learning is a kind of behaviour cloning method, which requires an expert to show the learner it's behaviour, in our case, given an image from the duckietown environment, what action would the expert do. The images are the inputs of our model, and the actions (velocity, steering) are the desired outputs. First, we fitted the model using the previously collected data, using a convolutional neural network (the implementation can be found in ./duckieGym/model.py), this resulted in a 0.3 mse validation loss model. The main drawback of imitation learning is that the model is only trained on perfect conditions, there are no examples of leaving the road and coming back from the grass, thus this alone is not enough. Hence we decided to extend the training DAgger, which is an algorithm solving our problem by letting the model go around in the environment, and when the model is not performing well, the expert takes the control back, showing the model what it should do in such cases. We collect every observation of this process, then the model will be further trained using them.

Data 1 contain Training Data for Lane Following and Data 2 for Pedestrians.
These data gathered by ourself. We extracted these from DuckieTown’s own simulation environment, half with manual guidance and half with pre-written automation.

The script automatic.py runs a simulation in the duckieTown environment and saves the images in an original form to an "/originalImages" folder and also does some preprocessing on the images that includes resizing and a little color manipulation. These smaller images are saved to the "/preprocessedImages" folder. The corresponding labels to the images are also saved to a text file called "my_app.txt". Each row contains an integer, and two floats describing the image ID, the velocity and the steering to that particular image.

To start training run model.py. This script reads the data from the "/preprocessedImages" folder. The data is then scaled and a model is created. After fitting the model to the training data with a validation split the model is automatically evaluated with the test split that had also been created. This prints an eval score to the console. After this, all the predicted and the real values are displayed for the test split. Each row in the console contains 4 numbers in the form of
\[pred_vel, pred_steer\], \[y_vel, y_steer\].
These numbers may look odd at first, however these are not the final predictions since the Y labels have been scaled with a standardScaler and the printed results will have to be scaled back to have a meaning.
For demonstrational purposes, our best model, to-date, can also be downloaded from:
https://onedrive.live.com/?authkey=%21AP7HuJgjv7pjAS4&id=7961F412AD7C6165%211597&cid=7961F412AD7C6165

A Dagger algorithm has also been implemented. The learner can be found in the DaggerLearner.py file, which contains a wrapper class for our model. The Teacher is implemented in the DaggerTeacher.py file. This implemenation is strongly based on the code that can be found in the automatic.py file, that generates the original data. The dagger implementation can be found in the DAgger.py and this is the file that you have to run in order to start the algorithm. This creates a teacher, a learner, a duckieTown environment and starts the process. The generated images are saved to the "/daggerObservations" directory and this fodler also contains the "labels.txt" that are the labels for the generated images. The structure of this text file is the same as that of the "my_app.txt" file

## Milestone 3:
The required software environment can be found here: https://github.com/Marci0707/DuckieProject/blob/main/duckieGym/requirements.txt

The `automatic.py` file is used for generating driving images via an automatic agent.
More driving data can be produced with using `human.py` by using WASD controls (originally with joystick).
A callback class `LoggerCallback` has been implemented to log the learning procedure. The data provided by it may help other research in the future.
The learning itself is done in the `model.py`'s *train_model* function.
The algorithm of DAgger is implemented in the `dagger.py`. It uses previously processed images loaded by `data_reader.py`.
The optimization of hyperparameters is done in the `hyper_optimization.py`.


Our best performing model can be downloaded from here:
https://www.mediafire.com/file/h2i7zxxw2n8y5lj/Balra%25C3%2589sJobbraIsKanyarodik.zip/file
It is strongly advised to multply the predicted velocity and steering by a positive number. We got the best results by multiplying the predicted velocity by 1.6 and multiplying the predicted steering by 10.0 and then use these values when stepping the environment.This is the final form of the model with which we managed to make the DEMO video linked at the start of README.md (https://youtu.be/8hvmge7oYk4)
# Execution:

Plain model:
```toc
python3 model.py
```
Hyperoptimization:
```toc
python3 hyper_optimization.py
```
DAgger:
```toc
python3 dagger.py (--map-name "name")  --model_path "path"
```
Simulator:
```toc
python3 modelsimulator.py (--map-name "name")  --model_path "path"
```

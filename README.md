# Introduction
For our group project we will be making a music generation application utilizing machine learning. Prior research has been conducted in this area with researchers experimenting with which different neural network architectures produces the desired output music. Hadjeres et al. proposed an architecture named Anticipation-RNN composed primarily of two recurrent neural networks that featured unary constraints on notes (2017). Briot & Pachet address some of the main challenges involved in deep learning for music generation including aspects of user control, lack of incremental development found in human music creation, and tendencies for memorization/plagiarism (2020). Sturm et al tested different machine learning models for music generation and found that the models often generated music that sounded much different from the music from the models’ training data, which, while a limitation, showed that the training data these models used was just one of many factors going into the generated output (2019). For our dataset, we plan on utilizing the [NES music database](https://github.com/chrisdonahue/nesmdb) for training our model. The dataset contains several hundred different soundtracks formatted as MIDI files that follow the 8-bit genre. We plan to derive 3 features from our dataset: pitch, step, and duration which will contribute to the type of music our model will generate.

# Problem definition
One of the most important aspects of the user experience in video games is music. However, most indie and individual game developers do not have a musical background. So, our goal is to provide an accessible tool for game developers to enhance their games with music, regardless of their musical experience or background.

# Data Collection
The music database we are working with includes a large amount of audio files that are sound effects. We cleaned our dataset by removing these files to ensure that our dataset contains only music soundtracks. We differentiated between sound effect and song by setting a length threshold of 30 seconds and removing all sound files that are less than this threshold. Files with a length greater than 30 seconds were kept as part of our training dataset.

# Methods
We will use Tensorflow as our primary library for creating our music generator. Music from the NES music database will be ingested as MIDI files and processed with PrettyMIDI. The data is stored in a MIDI format and was already cleaned to begin with. MIDI isn’t a playable audio source, it is a source of musical information such as pitch, timing, etc. that tells other devices what to play depending on the musical library it has. The dataset we’re using is already clean to begin with but we are processing the information to tailor fit our purposes. The dataset includes information like note, velocity, and timbre but we’re going to be focusing on pitch duration and step for the model. Notes will be extracted along with their respective pitch, step, and duration and converted into a training dataset. A model will be created with LSTM and dense layers using Adam as the optimizer and Tensorflow’s SparseCategoricalCrossentropy as the loss function. The resulting model generates musical notes in a sequence with a given temperature variable that controls the randomness of notes.


# References
Briot, JP., Pachet, F. Deep learning for music generation: challenges and directions. Neural Comput & Applic 32, 981–993 (2020). https://doi.org/10.1007/s00521-018-3813-6

Hadjeres, G., & Nielsen, F. (2017). Interactive music generation with positional constraints using 
anticipation-rnns. arXiv preprint arXiv:1709.06404.

Bob L. Sturm, Oded Ben-Tal, Úna Monaghan, Nick Collins, Dorien Herremans, Elaine Chew, Gaëtan Hadjeres, Emmanuel Deruty & François Pachet (2019) Machine learning research that matters for music creation: A case study, Journal of New Music Research, 48:1, 36-55, DOI: 10.1080/09298215.2018.1515233 

# Results and Discussion
The potential result would be a 30 second clip of 8-bit music generated by the model. By using the NES Music dataset to train the model, we aim to produce 8-bit music that is original and distinct. Due to the subjective nature of music, it would be extremely difficult to have quantitative metrics to evaluate our results. We will instead need to lean on a more qualitative evaluation.

# Timeline
Below is a link to our Gantt chart which organizes our projects timeline
https://1drv.ms/x/s!Am9-lOMC68OKawyaCieqDdI-cNA

# Contributions table
| Name | Disha | Cason | Emily | William | Derek |
| ---- | ----- | ----- | ----- | ------- | ----- |
| Task completed | research for introduction, powerpoint, exploring datasets | Create github pages, problem definition, references, powerpoint | Added MusPy Source, Filmed and edited video | Found Briot source, added information to introduction, Potential Results, Part of powerpoint | Added Gantt Chart, identified model architecture, added paper to references, worked on introduction |

# Checkpoint
By midterm report:
At minimum, we aim to have a cleaned version of our [dataset](https://github.com/chrisdonahue/nesmdb) that includes 8-bit songs that are easily interpretable for our model. Tensorflow should be set up in our repository and the set of songs we have should be loaded in. Our goal also includes breaking each song into individual notes and “separating” each note into 3 different groups: step, duration, and pitch. At this point, all data-related needs should be handled with the only remaining task being setting up a model that can generate songs based on our song dataset. 
If we are able to finalize our dataset and have remaining time, we’ll also aim to set up a prototype version of our model. It should be able to generate songs, but doing so at a high quality is not our goal.
<br />
By final report:
At the end of our project, we will have a completed model. If we are not able to develop a prototype before the midterm report, our priority will be to develop a basic working version. By the end, our goal will be to optimize our existing model to generate higher quality songs with more coherent notes and a distinct melody. 

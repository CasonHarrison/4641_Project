# Team 25 4641 Proposal

# Introduction
For our group project we will be making a music generation application utilizing machine learning. Prior research has been conducted in this area with researchers experimenting with which different neural network architectures produces the desired output music. Hadjeres et al. proposed an architecture named Anticipation-RNN composed primarily of two recurrent neural networks that featured unary constraints on notes. Briot & Pachet address some of the main challenges involved in deep learning for music generation including aspects of user control, lack of incremental development found in human music creation, and tendencies for memorization/plagiarism. Sturm et al tested different machine learning models for music generation and found that the models often generated music that sounded much different from the music from the models’ training data, which, while a limitation, showed that the training data these models used was just one of many factors going into the generated output (2019). We plan on primarily utilizing the NES music database for training our model along with the MIDI dataset and Million Song dataset as secondary data sources. The dataset’s features include variables such as pitch, duration, and step which .

# Problem definition
One of the most important aspects of the user experience in video games is music. However, most indie and individual game developers do not have a musical background. So, our goal is to provide an accessible tool for game developers to enhance their games with music, regardless of their musical experience or background.
Methods
We will use Tensorflow as our primary library for creating our music generator. Music from the NES music database will be ingested as MIDI files and processed with PrettyMIDI. Notes will be extracted along with their respective pitch, step, and duration and converted into a training dataset. A model will be created with LSTM and dense layers using Adam as the optimizer and Tensorflow’s SparseCategoricalCrossentropy as the loss function. The resulting model generates musical notes in a sequence with a given temperature variable that controls the randomness of notes.

# References
Briot, JP., Pachet, F. Deep learning for music generation: challenges and directions. Neural Comput & Applic 32, 981–993 (2020). https://doi.org/10.1007/s00521-018-3813-6
Hadjeres, G., & Nielsen, F. (2017). Interactive music generation with positional constraints using 
anticipation-rnns. arXiv preprint arXiv:1709.06404.
Bob L. Sturm, Oded Ben-Tal, Úna Monaghan, Nick Collins, Dorien Herremans, Elaine Chew, Gaëtan Hadjeres, Emmanuel Deruty & François Pachet (2019) Machine learning research that matters for music creation: A case study, Journal of New Music Research, 48:1, 36-55, DOI: 10.1080/09298215.2018.1515233 
Donahue, C., Mao, H. H., &amp; McAuley, J. (2018). The NES Music Database: A Multi-Instrumental Dataset with Expressive Performance Attributes. https://doi.org/https://doi.org/10.48550/arXiv.1806.04278 
Dong, H. W., Chen, K., McAuley, J., & Berg-Kirkpatrick, T. (2020). MusPy: A toolkit for symbolic music 	generation. arXiv preprint arXiv:2008.01951. https://arxiv.org/abs/2008.01951
 
#Potential Results
The potential result would be a 30 second clip of 8-bit music generated by the model. By using the NES Music dataset to train the model, we aim to produce 8-bit music that is original and distinct. Due to the subjective nature of music, it would be extremely difficult to have quantitative metrics to evaluate our results. We will instead need to lean on a more qualitative evaluation.


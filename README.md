# JDR-L
Dataset:

Original dataset is stored in Data folder

Embedding:

user-item embedding is implemented with item2vec, the origin code can be found in https://github.com/lujiaying/MovieTaster-Open

item2vec Reference:

Barkan, Oren, and Noam Koenigstein. "Item2vec: neural item embedding for collaborative filtering." Machine Learning for Signal Processing (MLSP), 2016 IEEE 26th International Workshop on. IEEE, 2016.

JayveeHe, https://github.com/JayveeHe/MusicTaster. Github.

Usage:

To get the user-item embedding, run mpr-lstm.py with prepare_dataset()

To run the proposed JDR model, run mpr-lstm.py with run()

note that: it is extremely important to seek a proper user-item embedding for model training stage. That is to say, prepare_dataset() and run() should be fine truning together to get the best model performance. 

Two contributing method ：

Wang Z, Zhang Y, Chang C Y. Integrating order information and event relation for script event prediction[C]//Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017: 57-67.

code：https://github.com/wangzq870305/event_chain

Yu L, Zhang C, Liang S, et al. Multi-order Attentive Ranking Model for Sequential Recommendation[C]//Proceedings of the AAAI. 2019.

code：https://github.com/voladorlu/MARank.

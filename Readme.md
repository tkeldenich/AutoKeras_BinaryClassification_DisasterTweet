# **Utiliser AutoKeras pour de la Classification Binaire** - [Lire l'article](https://inside-machinelearning.com/autokeras-la-librairie-du-futur/)

## **Charger les données**

Dans ce programme, on va se servir d'**AutoKeras** pour faire de la **Classification de Texte**. Pour cela, on utilise les données de la **compétition Kaggle** *Natural Language Processing with Disaster Tweets*.

L'objectif est de **classifier des tweets** : parlent-ils de **catastrophes** entrain de se passer ou bien de **la vie de tous les jours ?**

Notre algorithmes de **Deep Learning** devra en décider **par lui-même !**

Ici on a donc des **données textes à classifier** en **1** (catastrophe) ou en **0** (vie quotidienne).

On commence par importer les **librairies de base** pour faire du **Machine Learning :**
- numpy
- pandas


```python
import numpy as np
import pandas as pd
```

On importe les **tweets** qui se trouvent au **format CSV** sur **Github** à [cette adresse](https://github.com/tkeldenich/AutoKeras_BinaryClassification_DisasterTweet).


```python
!git clone https://github.com/tkeldenich/AutoKeras_BinaryClassification_DisasterTweet.git  &> /dev/null
```

Ensuite on **charge** les données de **train** et de **test**.

Ici, une **différence** par rapport à d'habitude, pour utiliser AutoKeras nous devons **transformer notre liste** de tweet en **array numpy**, pour cela on utilise la fonction *to_numpy()*.


```python
train_data = pd.read_csv('/content/AutoKeras_BinaryClassification_DisasterTweet/train.csv', index_col = 'id')
train_data = train_data.reset_index(drop = True)

X_train = train_data[['text']].to_numpy()
y_train = train_data[['target']].to_numpy()
```


```python
test_data = pd.read_csv('/content/AutoKeras_BinaryClassification_DisasterTweet/test.csv')

test_id = test_data[['id']]

X_test = test_data[['text']].to_numpy()
```

On peut **vérifier** que nos données sont bien sous forme d'**array numpy** :


```python
X_train
```




    array([['Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all'],
           ['Forest fire near La Ronge Sask. Canada'],
           ["All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected"],
           ...,
           ['M1.94 [01:04 UTC]?5km S of Volcano Hawaii. http://t.co/zDtoyd8EbJ'],
           ['Police investigating after an e-bike collided with a car in Little Portugal. E-bike rider suffered serious non-life threatening injuries.'],
           ['The Latest: More Homes Razed by Northern California Wildfire - ABC News http://t.co/YmY4rSkQ3d']],
          dtype=object)



### **Modèle AutoKeras**

Pour **utiliser AutoKeras** la première chose à faire est d'**installer la librairie** sur notre serveur :


```python
!pip install autokeras &> /dev/null
```

On **importe** ensuite la **librairie**.


```python
import autokeras as ak
```

La **partie intéressante** commence ! On veut faire de la **classification de texte**, on utilise donc la fonction *TextClassifier()* de **AutoKeras**.

Cette fonction possède un **paramètre** principale : **max_trials**.

**max_trials** permet de déterminer **le nombre de modèle** que AutoKeras va tester avant de **choisir le meilleur d'entre eux**.

**D'autres paramètres existe** que vous pouvez **consulter** sur la [documentation](https://autokeras.com/text_classifier/).


```python
clf = ak.TextClassifier(
    max_trials=3
)
```

Puis, on **entraîne notre modèle !**


```python
clf.fit(X_train, y_train, validation_split = 0.2, epochs=4)
```

    Trial 3 Complete [00h 12m 40s]
    val_loss: 0.38932037353515625
    
    Best val_loss So Far: 0.38932037353515625
    Total elapsed time: 00h 13m 58s
    INFO:tensorflow:Oracle triggered exit
    Epoch 1/4
    238/238 [==============================] - 195s 752ms/step - loss: 0.6609 - accuracy: 0.6415
    Epoch 2/4
    238/238 [==============================] - 178s 751ms/step - loss: 0.4111 - accuracy: 0.8295
    Epoch 3/4
    238/238 [==============================] - 178s 751ms/step - loss: 0.3263 - accuracy: 0.8710
    Epoch 4/4
    238/238 [==============================] - 178s 751ms/step - loss: 0.2626 - accuracy: 0.9006


    WARNING:absl:Found untraced functions such as word_embeddings_layer_call_fn, word_embeddings_layer_call_and_return_conditional_losses, position_embedding_layer_call_fn, position_embedding_layer_call_and_return_conditional_losses, type_embeddings_layer_call_fn while saving (showing 5 of 945). These functions will not be directly callable after loading.
    WARNING:absl:Found untraced functions such as word_embeddings_layer_call_fn, word_embeddings_layer_call_and_return_conditional_losses, position_embedding_layer_call_fn, position_embedding_layer_call_and_return_conditional_losses, type_embeddings_layer_call_fn while saving (showing 5 of 945). These functions will not be directly callable after loading.


    INFO:tensorflow:Assets written to: ./text_classifier/best_model/assets


    INFO:tensorflow:Assets written to: ./text_classifier/best_model/assets


**Simple, rapide, efficace...** que demande le peuple ?

On réalise ensuite notre **prédiction**.


```python
predictions = clf.predict(X_test)
```

    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._config_dict


    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._config_dict


    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._config_dict.initializer


    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._config_dict.initializer


    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._config_dict.initializer.config


    WARNING:tensorflow:Unresolved object in checkpoint: (root).model._config_dict.initializer.config


    WARNING:tensorflow:A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.


    WARNING:tensorflow:A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.


Cette **prédiction** est **composé de 1 et de 0** au format **float**, on les transforme en **int**.


```python
predictions = list(map(int, predictions))
```

**3 lignes de code** pour réaliser le **preprocessing, l'entraînement et la prédiction**. On peut **difficilement faire mieux !**

Et pour **exporter le modèle** et le **réutiliser ailleurs**, voilà **la marche à suivre :**


```python
model = clf.export_model()


try:
    model.save("model_autokeras", save_format="tf")
except Exception:
    model.save("model_autokeras.h5")
```

    WARNING:absl:Found untraced functions such as word_embeddings_layer_call_fn, word_embeddings_layer_call_and_return_conditional_losses, position_embedding_layer_call_fn, position_embedding_layer_call_and_return_conditional_losses, type_embeddings_layer_call_fn while saving (showing 5 of 945). These functions will not be directly callable after loading.
    WARNING:absl:Found untraced functions such as word_embeddings_layer_call_fn, word_embeddings_layer_call_and_return_conditional_losses, position_embedding_layer_call_fn, position_embedding_layer_call_and_return_conditional_losses, type_embeddings_layer_call_fn while saving (showing 5 of 945). These functions will not be directly callable after loading.


    INFO:tensorflow:Assets written to: model_autokeras/assets


    INFO:tensorflow:Assets written to: model_autokeras/assets


Ainsi que les étapes pour **réutiliser ce modèle exporter** :


```python
#loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)
#prediction = loaded_model.predict(X_test)
```

Et voilà, nous avons vu la **base d'AutoKeras** et il ne vous reste plus qu'à l'**utiliser à votre guise.**

N'hésitez pas à **faire un tour** sur [la documentation d'AutoKeras](https://autokeras.com) pour en **apprendre plus.**

Cette librairie est un **petit bijoux** et n'annonce que du bon pour **le futur du Machine Learning !** ;)

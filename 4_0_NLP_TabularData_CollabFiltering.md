# Lesson 4

[Video](https://youtu.be/C9UdVPE3ynA) / [Lesson Forum](https://forums.fast.ai/t/lesson-4-official-resources-and-updates/30317)

## Technical  resources
- [Brief lesson notes from @boy1729](https://medium.com/@boy1729/deep-learning-ver3-lesson-4-8f085a1e28ca)
- [Lesson 4 in-class discussion](https://forums.fast.ai/t/lesson-4-in-class-discussion/30318)
- [Lesson 4 advanced discussion](https://forums.fast.ai/t/lesson-4-advanced-discussion/30319)

## Readings 
- [QCon.ai keynote on Analyzing & Preventing Unconscious Bias in Machine Learning](https://www.infoq.com/presentations/unconscious-bias-machine-learning)
- [PyBay keynote with case studies of what can go wrong, and steps toward solutions](https://www.youtube.com/watch?v=WC1kPtG8Iz8&list=PLtmWHNX-gukLQlMvtRJ19s7-8MrnRV6h6)
- [Workshop on Word Embeddings, Bias in ML, Why You Don't Like Math, & Why AI Needs You](https://www.youtube.com/watch?v=25nC0n9ERq4)
- [AI Ethics Resources (includes links to experts to follow)](https://www.fast.ai/2018/09/24/ai-ethics-resources/)
- [What HBR Gets Wrong About Algorithms and Bias](http://www.fast.ai/2018/08/07/hbr-bias-algorithms/)
- [When Data Science Destabilizes Democracy and Facilitates Genocide](http://www.fast.ai/2017/11/02/ethics/)
- [Vim and Ctags for fast function definition lookup](https://andrew.stwrt.ca/posts/vim-ctags/)

We're going to do a deeper dive into NLP transfer learning today. We're going to then look at tabular data and  collaborative filtering which are both super useful applications. 

## Natural Language Processing (NLP) [[2:00](https://youtu.be/C9UdVPE3ynA?t=120)]

### A quick review

NLP is natural language processing. It's about taking text and doing something with it. Classifying a text or classifying a document can be used for anything from:

- Spam prevention 
- Identifying fake news
- Finding a diagnosis from medical reports
- Finding mentions of your product in Twitter 


![](lesson4/poster.pdf)


This series of three steps that you see here (and I'm sure you recognize this classification matrix) is what we're going to start by digging into.

![](lesson4/1.png)

We're going to start out with a movie review like this one and decide whether it's positive or negative sentiment about the movie. We have, in the training set, 25,000 movie reviews and for each one we have information: they liked it, or they didn't like it. Our neural networks (remember, they're just a bunch of matrix multiplies and simple nonlinearities﹣particularly replacing negatives with zeros), those weight matrices start out random. So if you start out with with some random parameters and try to train those parameters to learn how to recognize positive vs. negative movie reviews, you literally have 25,000 ones and zeros to actually tell you I like this one I don't like that one. That's clearly not enough information to learn, basically, how to speak English﹣how to speak English well enough to recognize they liked this or they didn't like this. Sometimes that can be pretty nuanced. Particularly with movie reviews because these are like online movie reviews on IMDB, people can often use sarcasm. It could be really quite tricky. 

Until very recently, in fact, this year, neural nets didn't do a good job at all of this kind of classification problem. And that was why﹣there's not enough information available. So the trick, hopefully you can all guess, is to use transfer learning. . 

### Transfer learning in NLP [[6:04](https://youtu.be/C9UdVPE3ynA?t=364)]

We're going to start with a pre-trained model. e.g. for ImageNet, that was originally built as a model to predict which of a thousand categories each photo falls into, and can be fine-tuned for all kinds of different things. So we're going to start with a pre-trained model called a language model.

A language model has a very specific meaning in NLP: a model that learns to predict the next word of a sentence. 
Here's your language model and it has read:

- "I'd like to eat a hot ___": Obviously, "dog", right?  
- "It was a hot ___": Probably "day"

Previous approaches to NLP use something called n-grams largely which predict how often do these pairs or triplets of words tend to appear next to each other. They don't perform well, as there's not enough information here to decide what the next word probably is. But with a neural net, you can. 

If you train a neural net to predict the next word of a sentence then you have a lot of information. Rather than having a single bit to every 2,000 word movie review: "liked it" or "didn't like it", every single word, you can try and predict the next word. So in a 2,000 word movie review, there are 1,999 opportunities to predict the next word. Better still, you don't just have to look at movie reviews. We started with Wikipedia. 

#### Wikitext 103 [[8:30](https://youtu.be/C9UdVPE3ynA?t=510)]

Stephen Merity and some of his colleagues built something called Wikitext 103 dataset which is simply a subset of most of the largest articles from Wikipedia with a little bit of pre-processing that's available for download. I built a neural net which would predict the next word in every significantly sized Wikipedia article. It's something like a billion tokens. So we've got a billion separate things to predict. Every time we make a mistake on one of those predictions, we get the loss, we get gradients from that, and we can update our weights, and they can better and better until we can get pretty good at predicting the next word of Wikipedia.

At that point, I've got a model that knows probably how to complete sentences like this, so it knows quite a lot about English and quite a lot about how the world works﹣what kinds of things tend to be hot in different situations, for instance. Ideally, it would learn things like "in 1996 in a speech to the United Nations, United States president _____ said "... Now that would be a really good language model, because it would actually have to know who is this United States president in that year. So getting really good at training language models is a great way to teach a neural-net a lot about what is our world, what's in our world, how do things work in our world. It's a really fascinating topic, and it's actually one that philosophers have been studying for hundreds of years now. 

We've made available a pre-trained Wikitext model for NLP and we can do transfer learning to create a new language model that's specifically good at predicting the next word of movie reviews.

#### Fine-tuning Wikitext to create a new language model [[11:10](https://youtu.be/C9UdVPE3ynA?t=670)] 

If we can build a language model that's good at predicting the next word of movie reviews pre-trained with the Wikitext model, its going to understand a lot about "my favorite actor is Tom ____. " Or "I thought the photography was fantastic but I wasn't really so happy about the _____ (director)." It's going to learn a lot about specifically how movie reviews are written. It'll even learn things like what are the names of some popular movies.

That means we can still use a huge corpus of lots of movie reviews even if we don't know whether they're positive or negative to learn a lot about how movie reviews are written. So for all of this pre-training and all of this language model fine-tuning, we don't need any labels at all. It is what the researcher Yann LeCun calls **self supervised learning**. In other words, it's a classic supervised model﹣we have labels, but the labels are not things that somebody else have created. They're built into the dataset itself. At this point, we've now got something that's good at understanding movie reviews and we can fine-tune that with transfer learning to classify movie reviews to be positive or negative. So my hope was (when I tried this last year) that at that point, 25,000 ones and zeros would be enough feedback to fine-tune that model and it turned out it absolutely was.

**Question**: Does the language model approach works for text in forums that are informal English, misspelled words or slangs or shortforms like s6 instead of Samsung S 6? [[12:47](https://youtu.be/C9UdVPE3ynA?t=767)]

Yes, absolutely it does. Particularly if you start with your wikitext model and then fine-tune it with your "target" corpus. Corpus is just a bunch of documents (emails, tweets, medical reports, or whatever). You could fine-tune it so it can learn a bit about the specifics of the slang , abbreviations, or whatever that didn't appear in the full corpus. People thought that learning from something like Wikipedia wouldn't be that helpful because it's not that representative of how people tend to write. But turns out it's extremely helpful.

So language models themselves can be quite powerful. For example there was [a blog post](https://blog.swiftkey.com/swiftkey-debuts-worlds-first-smartphone-keyboard-powered-by-neural-networks/) from SwiftKey (the folks that do the mobile-phone predictive text keyboard) and they describe how they kind of rewrote their underlying model to use neural nets. This was a year or two ago. Now most phone keyboards seem to do this. You'll be typing away on your mobile phone, and in the prediction there will be something telling you what word you might want next. So that's a language model in your phone.

Another example was the researcher Andrej Karpathy who now runs all this stuff at Tesla, back when he was a PhD student, he created [a language model of text in LaTeX documents](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and created these automatic generation of LaTeX documents that then became these automatically generated papers.

#### Review of the basic process [[15:14](https://youtu.be/C9UdVPE3ynA?t=914)]

We've prepared a little IMDB sample that you can use which is in CSV file. You can read it in with Pandas and there's negative or positive, the text of each movie review, and boolean of is it in the validation set or the training set.

```python
path = untar_data(URLs.IMDB_SAMPLE)
path.ls()
```

```
[PosixPath('/home/jhoward/.fastai/data/imdb_sample/texts.csv'),
 PosixPath('/home/jhoward/.fastai/data/imdb_sample/models')]
```

```python
df = pd.read_csv(path/'texts.csv')
df.head()
```

|      | label    | text                                              | is_valid |
| ---- | -------- | ------------------------------------------------- | -------- |
| 0    | negative | Un-bleeping-believable! Meg Ryan doesn't even ... | False    |
| 1    | positive | This is a extremely well-made film. The acting... | False    |
| 2    | negative | Every once in a long while a movie will come a... | False    |
| 3    | positive | Name just says it all. I watched this movie wi... | False    |
| 4    | negative | This movie succeeds at being one of the most u... | False    |

So there's an example of a movie review:

```python
df['text'][1]
```

```
'This is a extremely well-made film. The acting, script and camera-work are all first-rate. The music is good, too, though it is mostly early in the film, when things are still relatively cheery. There are no really superstars in the cast, though several faces will be familiar. The entire cast does an excellent job with the script.<br /><br />But it is hard to watch, because there is no good end to a situation like the one presented. It is now fashionable to blame the British for setting Hindus and Muslims against each other, and then cruelly separating them into two countries. There is some merit in this view, but it\'s also true that no one forced Hindus and Muslims in the region to mistreat each other as they did around the time of partition. It seems more likely that the British simply saw the tensions between the religions and were clever enough to exploit them to their own ends.<br /><br />The result is that there is much cruelty and inhumanity in the situation and this is very unpleasant to remember and to see on the screen. But it is never painted as a black-and-white case. There is baseness and nobility on both sides, and also the hope for change in the younger generation.<br /><br />There is redemption of a sort, in the end, when Puro has to make a hard choice between a man who has ruined her life, but also truly loved her, and her family which has disowned her, then later come looking for her. But by that point, she has no option that is without great pain for her.<br /><br />This film carries the message that both Muslims and Hindus have their grave faults, and also that both can be dignified and caring people. The reality of partition makes that realisation all the more wrenching, since there can never be real reconciliation across the India/Pakistan border. In that sense, it is similar to "Mr & Mrs Iyer".<br /><br />In the end, we were glad to have seen the film, even though the resolution was heartbreaking. If the UK and US could deal with their own histories of racism with this kind of frankness, they would certainly be better off.'
```



So you can just go `TextDataBunch.from_csv` to grab a language model specific data bunch:

```python
data_lm = TextDataBunch.from_csv(path, 'texts.csv')
```

And then you can create a learner from that in the usual way and fit it.  

```python
data_lm.save()
```

You can save the data bunch which means that the pre-processing that is done, you don't have to do it again. You can just load it.

```python
data = TextDataBunch.load(path)
```



What happens behind the scenes if we now load it as a classification data bunch (that's going to allow us to see the labels as well)? 

```
data = TextClasDataBunch.load(path)
data.show_batch()
```

| text                                                         | label    |
| ------------------------------------------------------------ | -------- |
| xxbos xxfld 1 raising victor vargas : a review \n\n you know , raising victor vargas is like sticking your hands into a big , xxunk bowl of xxunk . it 's warm and gooey , but you 're not sure if it feels right . try as i might | negative |
| xxbos xxfld 1 now that che(2008 ) has finished its relatively short australian cinema run ( extremely limited xxunk screen in xxunk , after xxunk ) , i can xxunk join both xxunk of " at the movies " in taking steven soderbergh to task . \n\n it 's usually | negative |
| xxbos xxfld 1 many xxunk that this is n't just a classic due to the fact that it 's the first xxup 3d game , or even the first xxunk - up . it 's also one of the first xxunk games , one of the xxunk definitely the first | positive |
| xxbos xxfld 1 i really wanted to love this show . i truly , honestly did . \n\n for the first time , gay viewers get their own version of the " the bachelor " . with the help of his obligatory " hag " xxunk , james , a | negative |
| xxbos xxfld 1 this film sat on my xxunk for weeks before i watched it . i xxunk a self - indulgent xxunk flick about relationships gone bad . i was wrong ; this was an xxunk xxunk into the xxunk - up xxunk of new xxunk . \n\n the | positive |

As we described, it basically creates a separate unit (i.e. a "token") for each separate part of a word. So most of them are just for words, but sometimes if it's like an `'s` from `it's`, it will get its own token. Every bit of punctuation tends to get its own token (a comma, a full stop, and so forth). 

Then the next thing that we do is a numericalization which is where we find what are all of the unique tokens that appear here, and we create a big list of them. Here's the first ten in order of frequency:

```python
data.vocab.itos[:10]
```

```
['xxunk', 'xxpad', 'the', ',', '.', 'and', 'a', 'of', 'to', 'is']
```

And that big list of unique possible tokens is called the vocabulary which we just call it a "vocab". So what we then do is we replace the tokens with the ID of where is that token in the vocab:

```
data.train_ds[0][0]
```

```
Text xxbos xxfld 1 he now has a name , an identity , some memories and a a lost girlfriend . all he wanted was to disappear , but still , they xxunk him and destroyed the world he hardly built . now he wants some explanation , and to get ride of the people how made him what he is . yeah , jason bourne is back , and this time , he 's here with a vengeance .
```

```
data.train_ds[0][0].data[:10]
```

```
array([ 43,  44,  40,  34, 171,  62,   6, 352,   3,  47])
```

That's numericalization. As you'll learn, every word in our vocab is going to require a separate row in a weight matrix in our neural net. So to avoid that weight matrix getting too huge, we restrict the vocab to no more than (by default) 60,000 words.  And if a word doesn't appear more than two times, we don't put it in the vocab either. So we keep the vocab to a reasonable size in that way. When you see these `xxunk`, that's an unknown token. It just means this was something that was not a common enough word to appear in our vocab.

We also have a couple of other special tokens like (see `fastai.text.transform.py` for up-to-date info):

- `xxfld`: This is a special thing where if you've got like title, summary, abstract, body, (i. e. separate parts of a document), each one will get a separate field and so they will get numbered (e.g. `xxfld 2`).
- `xxup`: If there's something in all caps, it gets lower cased and a token called `xxup` will get added to it. 

#### With the data block API [[18:31](https://youtu.be/C9UdVPE3ynA?t=1111)]

Personally, I more often use the data block API because there's less to remember about exactly what data bunch to use, and what parameters and so forth, and it can be a bit more flexible. 

```python
data = (TextList.from_csv(path, 'texts.csv', cols='text')
                .split_from_df(col=2)
                .label_from_df(cols=0)
                .databunch())
```

So another approach to doing this is to just decide:

- What kind of list you're creating (i.e. what's your independent variable)? So in this case, my independent variable is text.
- What is it coming from? A CSV. 
- How do you want to split it into validation versus training? So in this case, column number two was the `is_valid` flag. 
- How do you want to label it? With positive or negative sentiment, for example. So column zero had that.
-  Then turn that into a data bunch.

That's going to do the same thing. 

```pythoin
path = untar_data(URLs.IMDB)
path.ls()
```

```
[PosixPath('/home/jhoward/.fastai/data/imdb/imdb.vocab'),
 PosixPath('/home/jhoward/.fastai/data/imdb/models'),
 PosixPath('/home/jhoward/.fastai/data/imdb/tmp_lm'),
 PosixPath('/home/jhoward/.fastai/data/imdb/train'),
 PosixPath('/home/jhoward/.fastai/data/imdb/test'),
 PosixPath('/home/jhoward/.fastai/data/imdb/README'),
 PosixPath('/home/jhoward/.fastai/data/imdb/tmp_clas')]
```

```python
(path/'train').ls()
```

```
[PosixPath('/home/jhoward/.fastai/data/imdb/train/pos'),
 PosixPath('/home/jhoward/.fastai/data/imdb/train/unsup'),
 PosixPath('/home/jhoward/.fastai/data/imdb/train/unsupBow.feat'),
 PosixPath('/home/jhoward/.fastai/data/imdb/train/labeledBow.feat'),
 PosixPath('/home/jhoward/.fastai/data/imdb/train/neg')]
```

Now let's grab the whole data set which has:

- 25,000 reviews in training 
- 25,000 interviews in validation
- 50,000 unsupervised movie reviews (50,000 movie reviews that haven't been scored at all) 

#### Language model [[19:44](https://youtu.be/C9UdVPE3ynA?t=1184)]

We're going to start with the language model. Now the good news is, we don't have to train the Wikitext 103 language model. Not that it's difficult﹣you can just download the wikitext 103 corpus, and run the same code. But it takes two or three days on a decent GPU, so not much point in you doing it. You may as well start with ours. Even if you've got a big corpus of like medical documents or legal documents, you should still start with Wikitext 103. There's just no reason to start with random weights. It's always good to use transfer learning if you can.

So we're gonna start fine-tuning our IMDB language model. 

```python
bs=48
```

```python
data_lm = (TextList.from_folder(path)
           #Inputs: all the text files in path
            .filter_by_folder(include=['train', 'test']) 
           #We may have other temp folders that contain text files so we only keep what's in train and test
            .random_split_by_pct(0.1)
           #We randomly split and keep 10% (10,000 reviews) for validation
            .label_for_lm()           
           #We want to do a language model so we label accordingly
            .databunch(bs=bs))
data_lm.save('tmp_lm')
```

We can say:

- It's a list of text files﹣the full IMDB actually is not in a CSV. Each document is a separate text file. 
- Say where it is﹣in this case we have to make sure we just to include the `train` and `test` folders.
- We randomly split it by 0.1. 

Now this is interesting﹣10%. Why are we randomly splitting it by 10% rather than using the predefined train and test they gave us? This is one of the cool things about transfer learning. Even though our validation set has to be held aside, it's actually only the labels that we have to keep aside. So we're not allowed to use the labels in the test set. If you think about in a Kaggle competition, you certainly can't use the labels because they don't even give them to you. But you can certainly use the independent variables. So in this case, you could absolutely use the text that is in the test set to train your language model. This is a good trick﹣when you do the language model, concatenate the training and test set together, and then just split out a smaller validation set so you've got more data to train your language model. So that's a little trick. 

So if you're doing NLP stuff on Kaggle, for example, or you've just got a smaller subset of labeled data, make sure that you use all of the text you have to train in your language model, because there's no reason not to.

- How are we going to label it? Remember, a language model kind of has its own labels. The text itself is labels so label for a language model (`label_for_lm`) does that for us.
- And create a data bunch and save it. That takes a few minutes to tokenize and numericalize. 



Since it takes some few minutes, we save it. Later on you can just load it. No need to run it again.

```python
data_lm = TextLMDataBunch.load(path, 'tmp_lm', bs=bs)
```

```python
data_lm.show_batch()
```

| idx  | text                                                         |
| ---- | ------------------------------------------------------------ |
| 0    | xxbos after seeing the truman show , i wanted to see the other films by weir . i would say this is a good one to start with . the plot : \n\n the wife of a doctor ( who is trying to impress his bosses , so he can get xxunk trying to finish her written course , while he s at work . but one day a strange man , who says that he s a plumber , tells her he s been called out to repair some pipes in there flat . |
| 1    | and turn to the wisdom of homeless people & ghosts . that 's a good plan . i would never recommend this movie ; partly because the sexual content is unnecessarily graphic , but also because it really does n't offer any valuable insight . check out " yentl " if you want to see a much more useful treatment of jewish tradition at odds with society . xxbos creep is the story of kate ( potente ) , an intensely unlikeable bourgeois bitch that finds herself somehow sleeping through the noise of the last |
| 2    | been done before but there is something about the way its done here that lifts it up from the rest of the pack . \n\n 8 out of 10 for dinosaur / monster lovers . xxbos i rented this movie to see how the sony xxunk camera shoots , ( i recently purchased the same camera ) and was blown away by the story and the acting . the directing , acting , editing was all above what i expected from what appeared at first glance to be a " low budget " type of |
| 3    | troubles . nigel and xxunk are the perfect team , i 'd watch their show any day ! i was so crushed when they removed it , and anytime they had it on xxup tv after that i was over the moon ! they put it on on demand one summer ( only the first eight episodes or so ) and i 'd spend whole afternoons watching them one after the other ... but the worst part ? it is now back on a channel called xxunk - and xxup it xxup 's xxup on |
| 4    | movie ! ) the movie is about edward , a obsessive - compulsive , nice guy , who happens to be a film editor . he is then lent to another department in the building , and he is sent to the posh yet violent world of sam campbell , the splatter and gore department . sam campbell , eddy 's new boss , is telling eddy about the big break on his movies , the gruesome loose limbs series , and he needs eddy to make the movie somewhat less violent so they can |



#### Training [[22:29](https://youtu.be/C9UdVPE3ynA?t=1349)]

At this point things are going to look very familiar. We create a learner:

```python
learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.3)
```

But instead of creating a CNN learner, we're going to create a language model learner. So behind the scenes, this is actually not going to create a CNN (a convolutional neural network), it's going to create an RNN (a recurrent neural network). We're going to be learning exactly how they're built over the coming lessons, but in short they're the same basic structure. The input goes into a weight matrix (i.e. a matrix multiply), that then you replace the negatives with zeros, and it goes into another matrix multiply, and so forth a bunch of times. So it's the same basic structure.

As usual, when we create a learner, you have to pass in two things:

- The data: so here's our language model data
- What pre-trade model we want to use: here, the pre-trade model is the Wikitext 103 model that will be downloaded for you from fastai if you haven't used it before just like ImageNet pre-trained models are downloaded for you.

This here (`drop_mult=0.3`) sets the amount of dropout. We haven't talked about that yet. We've talked briefly about this idea that there is something called regularization and you can reduce the regularization to avoid underfitting. So for now, just know that by using a number lower than one is because when I first tried to run this, I was under fitting. So if you reduced that number, then it will avoid under fitting.

Okay. so we've got a learner, we can `lr_find` and looks pretty standard:

```python
learn.lr_find()
```

```python
learn.recorder.plot(skip_end=15)
```

![](lesson4/n1.png)

Then we can fit one cycle.

```python
learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
```

```
Total time: 12:42
epoch  train_loss  valid_loss  accuracy
1      4.591534    4.429290    0.251909  (12:42)
```

What's happening here is we are just fine-tuning the last layers. Normally after we fine-tune the last layers, the next thing we do is we go `unfreeze` and train the whole thing. So here it is:

```python
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
```

```
Total time: 2:22:17
epoch  train_loss  valid_loss  accuracy
1      4.307920    4.245430    0.271067  (14:14)
2      4.253745    4.162714    0.281017  (14:13)
3      4.166390    4.114120    0.287092  (14:14)
4      4.099329    4.068735    0.292060  (14:10)
5      4.048801    4.035339    0.295645  (14:12)
6      3.980410    4.009860    0.298551  (14:12)
7      3.947437    3.991286    0.300850  (14:14)
8      3.897383    3.977569    0.302463  (14:15)
9      3.866736    3.972447    0.303147  (14:14)
10     3.847952    3.972852    0.303105  (14:15)
```

As you can see, even on a pretty beefy GPU that takes two or three hours. In fact, I'm still under fitting.  So probably tonight, I might train it overnight and try and do a little bit better. I'm guessing I could probably train this a bit longer because you can see the accuracy hasn't started going down again. So I wouldn't mind try to train that a bit longer. But the accuracy, it's interesting. 0.3 means we're guessing the next word of the movie review correctly about a third of the time. That sounds like a pretty high number﹣the idea that you can actually guess the next word that often. So it's a good sign that my language model is doing pretty well. For more limited domain documents (like medical transcripts and legal transcripts), you'll often find this accuracy gets a lot higher. So sometimes this can be even 50% or more. But 0.3 or more is pretty good.



#### Predicting with Language Model [[25:43](https://youtu.be/C9UdVPE3ynA?t=1543)]

You can now run `learn.predict` and pass in the start of a sentence, and it will try and finish off that sentence for you.

```python
learn.predict('I liked this movie because ', 100, temperature=1.1, min_p=0.001)
```

```
Total time: 00:10
```

```
'I liked this movie because  of course after yeah funny later that the world reason settings - the movie that perfect the kill of the same plot - a mention of the most of course . do xxup diamonds and the " xxup disappeared kill of course and the movie niece , from the care more the story of the let character , " i was a lot \'s the little performance is not only . the excellent for the most of course , with the minutes night on the into movies ( ! , in the movie its the first ever ! \n\n a'
```

Now I should mention, this is not designed to be a good text generation system. This is really more designed to check that it seems to be creating something that's vaguely sensible. There's a lot lot of tricks that you can use to generate much higher quality text﹣none of which we're using here. But you can kind of see that it's certainly not random words that it's generating. It sounds vaguely English like even though it doesn't make any sense.

At this point, we have a movie review model. So now we're going to save that in order to load it into our classifier (i.e. to be a pre-trained model for the classifier). But I actually don't want to save the whole thing. A lot of the second half of the language model is all about predicting the next word rather than about understanding the sentence so far. So the bit which is specifically about understanding the sentence so far is called the **encoder**, so I just save that (i.e. the bit that understands the sentence rather than the bit that generates the word).

```python
learn.save_encoder('fine_tuned_enc')
```



#### Classifier [[27:18](https://youtu.be/C9UdVPE3ynA?t=1638)]

Now we're ready to create our classifier. Step one, as per usual, is to create a data bunch, and we're going to do basically exactly the same thing:

```python
data_clas = (TextList.from_folder(path, vocab=data_lm.vocab)
             #grab all the text files in path
             .split_by_folder(valid='test')
             #split by train and valid folder (that only keeps 'train' and 'test' so no need to filter)
             .label_from_folder(classes=['neg', 'pos'])
             #remove docs with labels not in above list (i.e. 'unsup')
             .filter_missing_y()
             #label them all with their folders
             .databunch(bs=50))
data_clas.save('tmp_clas')
```

But we want to make sure that it uses exactly the same vocab that are used for the language model. If word number 10 was `the` in the language model, we need to make sure that word number 10 is `the` in the classifier. Because otherwise, the pre-trained model is going to be totally meaningless. So that's why we pass in the vocab from the language model to make sure that this data bunch is going to have exactly the same vocab. That's an important step.

`split_by_folder`﹣remember, the last time we had split randomly, but this time we need to make sure that the labels of the test set are not touched. So we split by folder.

And then this time we label it not for a language model but we label these classes (`['neg', 'pos']`). Then finally create a data bunch.

Sometimes you'll find that you ran out of GPU memory. I was running this in an 11G machine, so you should make sure this number (`bs`) is a bit lower if you run out of memory. You may also want to make sure you restart the notebook and kind of start it just from here (classifier section). Batch size 50 is as high as I could get on an 11G card. If you're using a p2 or p3 on Amazon or the K80 on Google, for example, I think you'll get 16G so you might be able to make this bit higher, get it up to 64. So you can find whatever batch size fits on your card.

So here is our data bunch:

```python
data_clas = TextClasDataBunch.load(path, 'tmp_clas', bs=bs)
data_clas.show_batch()
```

| text                                                         | label |
| ------------------------------------------------------------ | ----- |
| xxfld 1 match 1 : tag team table match bubba ray and spike dudley vs eddie guerrero and chris benoit bubba ray and spike dudley started things off with a tag team table match against eddie guerrero and chris benoit . according to the rules of the match , both | pos   |
| xxfld 1 i have never seen any of spike lee 's prior films , as their trailers never caught my interest . i have seen , and admire denzel washington , and jodie foster 's work , and have several of their dvds . i was , however , entirely | neg   |
| xxfld 1 pier paolo pasolini , or pee - pee - pee as i prefer to call him ( due to his love of showing male genitals ) , is perhaps xxup the most overrated european marxist director - and they are thick on the ground . how anyone can | neg   |
| xxfld 1 chris rock deserves better than he gives himself in " down to earth . " as directed by brothers chris & paul weitz of " american pie " fame , this uninspired remake of warren beatty 's 1978 fantasy " heaven can wait , " itself a rehash | neg   |
| xxfld 1 yesterday , i went to the monthly antique flea market that comes to town . i really have no interest in such things , but i went for the fellowship of friends who do have such an interest . looking over the hundreds of vendor , passing many | pos   |

```python
learn = text_classifier_learner(data_clas, drop_mult=0.5)
learn.load_encoder('fine_tuned_enc')
learn.freeze()
```

This time, rather than creating a language model learner, we're creating a text classifier learner. But again, same thing﹣pass in the data that we want, figure out how much regularization we need. If you're overfitting then you can increase this number (`drop_mult`). If you're underfitting, you can decrease the number. And most importantly, load in our pre train model. Remember, specifically it's this half of the model called the encoder which is the bit that we want to load in.

Then freeze, lr_find, find the learning rate and fit for a little bit.

```python
learn.lr_find()
```

```python
learn.recorder.plot()
```

![](lesson4/n2.png)

```python
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
```

```
Total time: 02:46
epoch  train_loss  valid_loss  accuracy
1      0.294225    0.210385    0.918960  (02:46)
```

We're already up nearly to 92% accuracy after less than three minutes of training. So this is a nice thing. In your particular domain (whether it be law, medicine, journalism, government, or whatever), you probably only need to train your domain's language model once. And that might take overnight to train well. But once you've got it, you can now very quickly create all kinds of different classifiers and models with that. In this case, already a pretty good model after three minutes. So when you first start doing this. you might find it a bit annoying that your first models take four hours or more to create that language model. But the key thing to remember is you only have to do that once for your entire domain of stuff that you're interested in. And then you can build lots of different classifiers and other models on top of that in a few minutes.

```python
learn.save('first')
```

```python
learn.load('first');
```

```python
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
```

```
Total time: 03:03
epoch  train_loss  valid_loss  accuracy
1      0.268781    0.180993    0.930760  (03:03)
```

We can save that to make sure we don't have to run it again. 

And then, here's something interesting. I'm not going to say `unfreeze`. Instead, I'm going to say `freeze_to`. What that says is unfreeze the last two layers, don't unfreeze the whole thing. We've just found it really helps with these text classification not to unfreeze the whole thing, but to unfreeze one layer at a time. 

- unfreeze the last two layers
- train it a little bit more
- unfreeze the next layer again
- train it a little bit more
- unfreeze the whole thing
- train it a little bit more

```python
learn.save('second')
```

```python
learn.load('second');
```

```python
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
```

```
Total time: 04:06
epoch  train_loss  valid_loss  accuracy
1      0.211133    0.161494    0.941280  (04:06)
```

```python
learn.save('third')
```

```python
learn.load('third');
```

```python
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
```

```
Total time: 10:01
epoch  train_loss  valid_loss  accuracy
1      0.188145    0.155038    0.942480  (05:00)
2      0.159475    0.153531    0.944040  (05:01)
```

You also see I'm passing in this thing (`moms=(0.8,0.7)`)﹣momentums equals 0.8,0.7. We are going to learn exactly what that means probably next week. We may even automate it. So maybe by the time you watch the video of this, this won't even be necessary anymore. Basically we found for training recurrent neural networks (RNNs), it really helps to decrease the momentum a little bit. So that's what that is.

That gets us a 94.4% accuracy after about half an hour or less of training. There's quite a lot less of training the actual classifier. We can actually get this quite a bit better with a few tricks. I don't know if we'll learn all the tricks this part. It might be next part. But even this very simple standard approach is pretty great.

![](lesson4/2.png)

If we compare it to last year's state of the art on IMDb, this is from The [CoVe paper](https://papers.nips.cc/paper/7209-learned-in-translation-contextualized-word-vectors.pdf) from McCann et al. at Salesforce Research. Their paper was 91.8% accurate. And the best paper they could find, they found a fairly domain-specific sentiment analysis paper from 2017, they've got 94.1%. And here, we've got 94.4%. And the best models I've been able to build aince have been about 95.1%. So if you're looking to do text classification, this really standardized transfer learning approach works super well.

### Tabular [[33:10](https://youtu.be/C9UdVPE3ynA?t=1990)]

![](lesson4/3.png)

So that was NLP. We'll be learning more about NLP later in this course. But now, I wanted to switch over and look at tabular. 

**Question**: Where does the magic number of <img src="https://latex.codecogs.com/gif.latex?2.6^{4}" title="2.6^{4}" /> in the learning rate come from? [[33:38](https://youtu.be/C9UdVPE3ynA?t=2018)]

```python
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
```

The learning rate is various things divided by 2.6 to the fourth. The reason it's to the fourth, you will learn about at the end of today. So let's focus on the 2.6. Basically, as we're going to see in more detail later today, this number, the difference between the bottom of the slice and the top of the slice is basically what's the difference between how quickly the lowest layer of the model learns versus the highest layer of the model learns. So this is called discriminative learning rates. So really the question is as you go from layer to layer, how much do I decrease the learning rate by? And we found out that for NLP RNNs, the answer is 2.6. 

How do we find out that it's 2.6? I ran lots and lots of different models using lots of different sets of hyper parameters of various types (dropout, learning rates,  and discriminative learning rate and so forth), and then I created something called a random forest which is a kind of model where I attempted to predict how accurate my NLP classifier would be based on the hyper parameters. And then I used random forest interpretation methods to basically figure out what the optimal parameter settings were, and I found out that the answer for this number was 2.6. So that's actually not something I've published or I don't think I've even talked about it before, so there's a new piece of information. Actually, a few months after I did this, Stephen Merity and somebody else did publish a paper describing a similar approach, so the basic idea may be out there already. 

Some of that idea comes from a researcher named Frank Hutter and one of his collaborators. They did some interesting work showing how you can use random forests to actually find optimal hyperparameters. A lot of people are very interested in this thing called Auto ML which is building models to figure out how to train your model, and then finding those rules of thumb.



### Tabular [[36:41](https://youtu.be/C9UdVPE3ynA?t=2201)]

Let's talk about tabular data, such as you might see in a spreadsheet, a relational database, or financial report, it can contain all kinds of different things. I tried to make a little list of some of the kinds of things that I've seen tabular data analysis used for:

![](lesson4/4.png)

Neural nets are extremely useful for tabular data. Although feature engineering doesn't go away, it certainly becomes simpler. Pinterest, for example, replaced the gradient boosting machines that they were using to decide how to put stuff on their homepage with neural nets. They presented at a conference this approach, and described how it really made engineering a lot easier because a lot of the hand created features weren't necessary anymore. You still need some, but it was just simpler. So they ended up with something that was more accurate, but perhaps even more importantly, it required less maintenance. I wouldn't say it's the only tool that you need in your toolbox for analyzing tabular data. But where I used to use random forests 99% of the time when I was doing machine learning with tabular data, I now use neural nets 90% of the time. It's my standard first go-to approach now, and it tends to be pretty reliable and effective.

One of the things that's made it difficult is that until now there hasn't been an easy way to create and train tabular neural nets. Nobody has really made it available in a library. So we've actually just created `fastai.tabular` and I think this is pretty much the first time that's become really easy to use neural nets with tabular data. 

#### Tabular examples [[39:51](https://youtu.be/C9UdVPE3ynA?t=2391)]

[lesson4-tabular.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson4-tabular.ipynb)

As usual, as well as importing fastai, import your application﹣so in this case, it's tabular.

```python
from fastai import *
from fastai.tabular import *
```

```python
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
```

We start with the data in a Pandas data frame, we've got an adult sample﹣it's a classic old dataset. If your data is in a relational database, Pandas can read from that. If it's in spark or Hadoop, Pandas can read from that. 

```python
dep_var = '>=50k'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [FillMissing, Categorify, Normalize]
```

```python
test = TabularList.from_df(df.iloc[800:1000].copy(), path=path, cat_names=cat_names, cont_names=cont_names)
```

```python
data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(800,1000)))
                           .label_from_df(cols=dep_var)
                           .add_test(test, label=0)
                           .databunch())
```

As usual, I think it's nice to use the data block API. So in this case, the list that we're trying to create is a tabular list and we're going to create it from a data frame. So you can tell it:

- What the data frame is. 
- What the path that you're going to use to save models and intermediate steps is.
- Then you need to tell it what are your categorical variables and what are your continuous variables. 


##### Processor [[45:04](https://youtu.be/C9UdVPE3ynA?t=2704)]

This is a lot like transforms in computer vision, which do things like flip a photo on its axis, turn it a bit, brighten it, or normalize it. A processor happens ahead of time, we basically pre-process the data frame rather than doing it as we go. So transformations are really for data augmentation﹣we want to randomize it and do it differently each time. Processes are the things that you want to do once, ahead of time.

```python
procs = [FillMissing, Categorify, Normalize]
```

We have a number of processes in the fastai library. And the ones we're going to use this time are: 

- `FillMissing`: Look for missing values and deal with them some way (replace it with the median and add a new column which is a binary column of saying whether that was missing or not) 
- `Categorify`: Find categorical variables and turn them into Pandas categories
- `Normalize` : Do a normalization ahead of time which is to take continuous variables and subtract their mean and divide by their standard deviation so they are zero-one variables.


For all of these things, whatever you do to the training set, you need to do exactly the same thing to the validation set and the test set. So whatever you replaced your missing values with, you need to replace them with exactly the same thing in the validation set. Fastai handles all these details for you. 

Then we're going to split into training versus validation sets. And in this case, we do it by providing a list of indexes so the indexes from 800 to a thousand, to keep your validation sets to be contiguous groups of things. If they're map tiles, they should be the map tiles that are next to each other, if their time periods, they should be days that are next to each other, if they are video frames, they should be video frames next to each other. So it's often a good idea to use `split_by_idx` and to grab a range that's next to each other if your data has some kind of structure like that or find some other way to structure it in that way.

We now need to add labels. In this case, the labels can come straight from the data frame we grabbed earlier, so we just have to tell it which column it is. 

```python
data.show_batch(rows=10)
```

| workclass        | education    | marital-status     | occupation        | relationship   | race  | education-num_na | age     | fnlwgt  | education-num | target |
| ---------------- | ------------ | ------------------ | ----------------- | -------------- | ----- | ---------------- | ------- | ------- | ------------- | ------ |
| Private          | Prof-school  | Married-civ-spouse | Prof-specialty    | Husband        | White | False            | 0.1036  | 0.9224  | 1.9245        | 1      |
| Self-emp-inc     | Bachelors    | Married-civ-spouse | Farming-fishing   | Husband        | White | False            | 1.7161  | -1.2654 | 1.1422        | 1      |
| Private          | HS-grad      | Never-married      | Adm-clerical      | Other-relative | Black | False            | -0.7760 | 1.1905  | -0.4224       | 0      |
| Private          | 10th         | Married-civ-spouse | Sales             | Own-child      | White | False            | -1.5823 | -0.0268 | -1.5958       | 0      |
| Private          | Some-college | Never-married      | Handlers-cleaners | Own-child      | White | False            | -1.3624 | 0.0284  | -0.0312       | 0      |
| Private          | Some-college | Married-civ-spouse | Prof-specialty    | Husband        | White | False            | 0.3968  | 0.4367  | -0.0312       | 1      |
| ?                | Some-college | Never-married      | ?                 | Own-child      | White | False            | -1.4357 | -0.7295 | -0.0312       | 0      |
| Self-emp-not-inc | 5th-6th      | Married-civ-spouse | Sales             | Husband        | White | False            | 0.6166  | -0.6503 | -2.7692       | 1      |
| Private          | Some-college | Married-civ-spouse | Sales             | Husband        | White | False            | 1.5695  | -0.8876 | -0.0312       | 1      |
| Local-gov        | Some-college | Never-married      | Handlers-cleaners | Own-child      | White | False            | -0.6294 | -1.5422 | -0.0312       | 0      |



```python
learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
```

```python
learn.fit(1, 1e-2)
```

```
Total time: 00:03
epoch  train_loss  valid_loss  accuracy
1      0.362837    0.413169    0.785000  (00:03)
```



[[52:12](https://youtu.be/C9UdVPE3ynA?t=3132)]

![](lesson4/5.png)


### Collaborative Filtering [[53:08](https://youtu.be/C9UdVPE3ynA?t=3188)]

Collaborative filtering is where you have information about who bought what, or who liked what﹣ it's something where you have something like a user, a reviewer, or whatever and information about what they've bought, what they've written about, or what they reviewed. In the most basic version of collaborative filtering, you just have two columns: something like user ID and movie ID and that just says this user bought that movie. So for example, Amazon has a really big list of user IDs and product IDs. Then you can add additional information to that table such as a review? So it's now user ID, movie ID, number of stars. You could add a timecode so  this user bought this product at this time and gave it this review. 

There are two ways you could draw that collaborative filtering structure. One is a two-column approach where you've got user and movie. And you've got user ID, movie ID﹣each pair basically describes that user watch that movie, possibly also number of stars (3, 4, etc). The other way you could write it would be you could have like all the users down here and all the movies along here.  You can look and find a particular cell in there to find out what could be the rating of that user for that movie, or there's just a `1` there if that user watched that movie, or whatever.

![](lesson4/whiteboard.gif)

So there are two different ways of representing the same information. Conceptually, it's often easier to think of it this way (the matrix on the right), but most of the time you won't store it that way. Explicitly because most of the time, you'll have what's called a very sparse matrix which is to say most users haven't watched most movies or most customers haven't purchased most products. So if you store it as a matrix where every combination of customer and product is a separate cell in that matrix, it's going to be enormous. So you tend to store it like the left or you can store it as a matrix using some kind of special sparse matrix format. [Rachel's computational linear algebra course](https://github.com/fastai/numerical-linear-algebra/blob/master/README.md) on fastai has lots of information about sparse matrix storage approaches. For now we're just going to kind of keep it in this format on left hand side.



[[56:38](https://youtu.be/C9UdVPE3ynA?t=3398)]

[lesson4-collab.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson4-collab.ipynb)

For collaborative filtering, there's a really nice dataset called MovieLens created by GroupLens group and you can download various different sizes (20 million ratings, 100,000 ratings). We've actually created an extra small version for playing around with which is what we'll start with today. And then probably next week, we'll use the bigger version. 

```python
from fastai import *
from fastai.collab import *
from fastai.tabular import *
```

You can grab the small version using `URLs.ML_SAMPLE`:

```python
user,item,title = 'userId','movieId','title'
```

```python
path = untar_data(URLs.ML_SAMPLE)
path
```

```
PosixPath('/home/jhoward/.fastai/data/movie_lens_sample')
```

```python
ratings = pd.read_csv(path/'ratings.csv')
ratings.head()
```

|      | userId | movieId | rating | timestamp  |
| ---- | ------ | ------- | ------ | ---------- |
| 0    | 73     | 1097    | 4.0    | 1255504951 |
| 1    | 561    | 924     | 3.5    | 1172695223 |
| 2    | 157    | 260     | 3.5    | 1291598691 |
| 3    | 358    | 1210    | 5.0    | 957481884  |
| 4    | 130    | 316     | 2.0    | 1138999234 |

It's a CSV so you can read it with Pandas. It's a list of user IDs﹣we don't actually know anything about who these users are. There's some movie IDs. There is some information about what the movies are, . Then there's the rating and the timestamp. We're going to ignore the timestamp for now. 

```python
data = CollabDataBunch.from_df(ratings, seed=42)
```

```python
y_range = [0,5.5]
```

```python
learn = collab_learner(data, n_factors=50, y_range=y_range)
```


That's all the data that we need. So you can now go ahead and say get `collab_learner` and you can pass in the data bunch. The minimum score is 0, the maximum score is 5. 


```python
learn.fit_one_cycle(3, 5e-3)
```

```
Total time: 00:04
epoch  train_loss  valid_loss
1      1.600185    0.962681    (00:01)
2      0.851333    0.678732    (00:01)
3      0.660136    0.666290    (00:01)
```

Now that you've got a learner, you can go ahead and call `fit_one_cycle` and trains for a few epochs, and there it is. So at the end of it, you now have something where you can pick a user ID and a movie ID, and guess whether or not that user will like that movie.

#### Cold start problem [[58:55](https://youtu.be/C9UdVPE3ynA?t=3535)]

The cold start problem is that the time you particularly want to be good at recommending movies is when you have a new user, and the time you particularly care about recommending a movie is when it's a new movie. But at that point, you don't have any data in your collaborative filtering system and it's really hard. 

We don't currently have anything built into fastai to handle the cold start problem and the only way I know of to solve it is to have a second model which is not a collaborative filtering model but a metadata driven model for new users or new movies.

I don't know if Netflix still does this, but certainly what they used to do when I signed up to Netflix was they started showing me lots of movies and saying "have you seen this?" "did you like it?" ﹣so they fixed the cold start problem through the UX. They found like 20 really common movies and asked me if I liked them, they used my replies to those 20 to show me 20 more that I might have seen, and by the time I had gone through 60,  there was no cold start problem anymore.

For new movies, it's not really a problem because like the first hundred users who haven't seen the movie go in and say whether they liked it, and then the next hundred thousand, the next million, it's not a cold start problem anymore. 

The other thing you can do if you can't go through that UX of asking people did you like those things (for example if you're selling products and you don't really want to show them a big selection of your products and say did you like this because you just want them to buy), is try and use a metadata based tabular model like what geography did they come from, their age and sex.

So collaborative filtering is specifically for once you have a bit of information about your users and movies or customers and products.

[[1:01:37](https://youtu.be/C9UdVPE3ynA?t=3697)]

**Question**: How does the language model trained in this manner perform on code switched data (Hindi written in English words), or text with a lot of emojis?

There's not many emojis in Wikipedia and where they are at Wikipedia it's more like a Wikipedia page about the emoji rather than the emoji being used in a sensible place. But you can (and should) do this language model fine-tuning where you take a corpus of text where people are using emojis in usual ways, so you fine-tune the Wikitext language model to your reddit or Twitter language model. And there aren't that many emojis. There are hundreds of thousands of possible words that people can be using, but a small number of possible emojis. So it'll very quickly learn how those emojis are being used.

I'm not really familiar with Hindi, but I'll take an example I'm very familiar with which is Mandarin. In Mandarin, you could have a model that's trained with Chinese characters. There are about five or six thousand Chinese characters in common use, but there's also a romanization of those characters called pinyin. It's a bit tricky because although there's a nearly direct mapping from the character to the pinyin (I mean there is a direct mapping but that pronunciations are not exactly direct),  there isn't direct mapping from the pinyin to the character because one pinyin corresponds to multiple characters. 

So the first thing to note is that if you're going to use this approach for Chinese, you would need to start with a Chinese language model. 

Actually fastai has something called [Language Model Zoo](https://forums.fast.ai/t/language-model-zoo-gorilla/14623) where we're adding more and more language models for different languages, and also increasingly for different domain areas like English medical texts or even language models for things other than NLP like genome sequences, molecular data, musical MIDI notes, and so forth.  So you would you obviously start there.

To then convert that (in either simplified or traditional Chinese) into pinyin, you could map the vocab directly. Or, since in multi-layer models it's only the first layer that converts the tokens into a set of vectors, you can actually throw that away and fine-tune just the first layer of the model. So that second part is going to require a few more weeks of learning before you exactly understand how to do that, but if this is something you're interested in doing, we can talk about it on the forum because it's a nice test of understanding.


### Collaborative filter with Microsoft Excel [[1:07:25](https://youtu.be/C9UdVPE3ynA?t=4045)]

[collab_filter.xlsx](https://github.com/fastai/course-v3/blob/master/files/xl/collab_filter.xlsx)

![](lesson4/6.png)

Microsoft Excel is one of my favorite ways to explore data and understand models. So here's some data, and one way to look at collaborative filtering data is like this:![](lesson4/7.png)

We grabbed from the MovieLens data the people that watched the most movies and movies that were the most watched, and limited the dataset down to those 15. When you do it that way, it's not sparse anymore. There's just a small number of gaps. This is something that we can now build a model with. We want to create something which can predict for user 293, will they like movie 49, for example. So we've got to come up with some function that can represent that decision. 

Here's a simple possible approach. We're going to take this idea of doing some matrix multiplications. Here's one matrix of random numbers (the left) and another matrix of random numbers (the top). More specifically, for each movie, I've created five random numbers, and for each user, I've created five random numbers.

![](lesson4/8.png)

Did user 14 like movie 27? We could multiply together the red vector and the purple vector to get a dot product. Then we can do that for every possible thing in here.This is the basic starting point of a neural net, you take the matrix multiplication of two matrices, and that's what your first layer always is. We can now use gradient descent to try to make these numbers (top) and these numbers (left) give us results that are closer to what we wanted. 

The next thing we need is a loss function. Movie 27 for user ID 14 should have been a rating of 3. With this random matrices, it's actually a rating of 0.91. So we can find the sum of squared errors <img src="https://latex.codecogs.com/gif.latex?(3-0.91)^{2}" title="(3-0.91)^{2}" /> and then we can add them up. So there's actually a sum squared in Excel already sum X minus y squared (`SUMXMY2`), so we can use just sum X minus y squared function, passing in those two ranges and then divide by the count to get the mean. 

![](lesson4/9.png)


#### Excel Solver [[1:14:30](https://youtu.be/C9UdVPE3ynA?t=4470)]

We have a loss, now we need to use gradient descent to try to modify our weight matrices to make that loss smaller. If you don't have solver, go to Excel Options → Add-ins, and enable "Solver Add-in". The gradient descent solver in Excel is called "Solver" and it just does normal gradient descent. You just go Data → Solver (you need to make sure that in your settings that you've enabled the solver extension which comes with Excel) and all you need to do is say which cell represents my loss function. So there it is, cell V41. The cells containing your variables are H19 to V23, and B25 to F39. You can say "okay, set your loss function to a minimum by changing those cells" and click on Solve: 

![](lesson4/10.png)

You'll see it starts at 2.81, then going down. And all that's doing is using gradient descent exactly the same way that we did when we did it manually in the notebook the other day. But it's rather than solving the mean squared error for `a@x`  in Python, instead it is solving the loss function here which is the mean squared error of the dot product of each of those vectors by each of these vectors.


Back to the [notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson4-collab.ipynb) [[1:17:02](https://youtu.be/C9UdVPE3ynA?t=4622)]


```python
data = CollabDataBunch.from_df(ratings, seed=42)
```

```
y_range = [0,5.5]
```

```python
learn = collab_learner(data, n_factors=50, y_range=y_range)
```

```python
learn.fit_one_cycle(3, 5e-3)
```

```
Total time: 00:04
epoch  train_loss  valid_loss
1      1.600185    0.962681    (00:01)
2      0.851333    0.678732    (00:01)
3      0.660136    0.666290    (00:01)
```

So over here we used `collab_learner` to get a model.

So in this case, the model contains:

- a set of weights for a user 
- a set of weights for an item 
- a set of biases for a user 
- a set of biases for an item 

And each one of those is coming from this thing called `embedding`. Here is the definition of `embedding`:

![](lesson4/12.png)



All it does is it calls this PyTorch thing called `nn.Embedding`. An embedding is a matrix of weights that looks something like this:

![](lesson4/13.png)

It's a matrix of weights which you can look up into, and grab one item out of it. You index into it as an array, and grab one vector out of it.In our case, we have an embedding matrix for a user and an embedding matrix for a movie. And here, we have been taking the dot product of them:

![](lesson4/14.png)

But if you think about it, that's not quite enough. Maybe there are certain movies that everybody likes more. Maybe there are some users that just tend to like movies more. So I  want to add a single number of how popular is this movie, and add a single number of how much does this user like movies in general. Those are called "bias" terms. Remember how I said there's this idea of bias and the way we dealt with that in our gradient descent notebook was we added a column of 1's. But what we tend to do in practice is we actually explicitly say I want to add a bias term. So we don't just want to have prediction equals dot product of these two things, we want to say it's the dot product of those two things plus a bias term for a movie plus a bias term for user ID.

#### Back to code [[1:23:55](https://youtu.be/C9UdVPE3ynA?t=5035)]

So that's basically what happens. We when we set up the model, we set up the embedding matrix for the users and the embedding matrix for the items. And then we also set up the bias vector for the users and the bias vector for the items.

![](lesson4/15.png)

When we calculate the model, we just multiply the two together. We take the dot product, add the bias, and (putting aside `y_range` for a moment) that's what we return. So you can see that our model is literally doing what we did in the spreadsheet with the tweak that we're also adding the bias. For these kinds of collaborative filtering problems, this kind of simple linear model tends to work pretty well.

Then there's one tweak that we do at the end which is that in our case we said that there's y range of between 0 and 5.5. So here's something to point out. So you do that dot product and you add on the two biases and that could give you any possible number along the number line from very negative through to very positive numbers. But we know that we always want to end up with a number between zero and five. What if we mapped that number line to this function. The shape of that function is called a sigmoid. And so, it's gonna asymptote to five and zero. 

![](lesson4/16.png)

Whatever number comes out of our dot product, we add the biases, then stick it through this function, it's never going to be higher than 5 and never going to be smaller than 0. 

```python
return torch.sigmoid(res) * (self.y_range[1]-self.y_range[0]) + self.y_range[0]
```

We take the result of this dot product plus biases, we put it through a sigmoid. A sigmoid is just a function which is basically<img src="https://latex.codecogs.com/gif.latex?\frac{1}{1&plus;e^{-x}}" title="\frac{1}{1+e^{-x}}" />, but the definition doesn't much matter. But it just has the shape that I just mentioned, and that goes between 0 and 1. If you then multiply that by `y_range[1]` minus `y_range[0]` plus `y_range[0]`, then that's going to give you something that's between `y_range[0]` and `y_range[1]`. 

The sigmoid at the end is a non-linearity, it only has one layer of weights. That actually turns out to give close to state-of-the-art performance. I've looked up online to find out like what are the best results people have on this MovieLens 100k database, and the results I get from this little thing is better than any of the results I can find from the standard commercial products that you can download that are specialized for this. And the trick seems to be that adding this little sigmoid makes a big difference.

[[1:29:09](https://youtu.be/C9UdVPE3ynA?t=5349)]


### Overview of important terminology [[1:31:24](https://youtu.be/C9UdVPE3ynA?t=5484)]

So what I wanted to close with today is, to take this collaborative filtering example and describe how we're going to build on top of it for the next three lessons to create the more complex neural networks we've been seeing. Roughly speaking, this is the bunch of concepts that we need to learn about:

- Inputs
- Weights/parameters
  - Random
- Activations
- Activation functions / nonlinearities
- Output
- Loss
- Metric
- Cross-entropy
- Softmax
- Fine tuning
  - Layer deletion and random weights
  - Freezing & unfreezing

Let's think about what happens when you're using a neural network to do image recognition. Let's take a single pixel. So you've got a red a green and a blue pixel. Each one of those is some number between 0 and 255, so red: 10, green: 20, blue 30. We basically treat that as a vector, and we multiply it by a matrix. So this matrix (depending on how you think of the rows and the columns), let's treat the matrix is having three rows and then how many columns? You get to pick. Just like with the collaborative filtering version, I decided to pick a vector of size five for each of my embedding vectors. So that would mean that's an embedding of size 5. You get to pick how big your weight matrix is. So let's make it size 5. This is 3 by 5.

Initially, this weight matrix contains random numbers. Remember we looked at `embedding` weight matrix just now?

![](lesson4/12.png)



There were two lines; the first line was create the matrix, and the second was fill it with random numbers? That's all we do. I mean it all gets hidden behind the scenes by fastai and PyTorch, but that's all it's doing. So it's creating a matrix of random numbers when you set it up. The number of rows has to be 3 to match the input, and the number of columns can be as big as you like. So after you multiply the input vector by that weight matrix, you're going to end up with a vector of size 5. 

People often ask how much linear algebra do I need to know to be able to do deep learning. This is the amount you need. And if you're not familiar with this, that's fine. You need to know about matrix products. You don't need to know a lot about them, you just need to know like computationally what are they and what do they do. You've got to be very comfortable with if a matrix of size blah times a matrix of size blah gives a matrix or size blah (i.e. how do the dimensions match up). So if you have 3, and they remember in numpy and PyTorch, we use @ times 3 by 5 gives a vector of size 5. 

Then what happens next; it goes through an activation function such as ReLU which is just `max(0,x)` and spits out a new vector which is, of course, going to be exactly the same size because **no activation function changes the size﹣it only changes the contents**. So that's still of size 5. 

What happens next? We multiply by another matrix. Again, it can be any number of columns, but the number of rows has to map nicely. So it's going to be 5 by whatever. Maybe this one has 5, let's say, by 10. That's going to give some output﹣it should be size 10 and again we put that through ReLU, and again that gives us something of the same size.

Then we can put that through another matrix.  Actually, just to make this a bit clearer (you'll see why in a moment), I'm going to use 8 not 10.

Let's say we're doing digit recognition. There are ten possible digits, so my last weight matrix has to be 10 in size. Because then that's going to mean my final output is a vector of 10 in size. Remember if you're doing that digit recognition, we take our actuals which is 10 in size. And if the number we're trying to predict was the number 3,  then that means that there is a 1 in the third position ([0,0,0,1,0,...]).

![](lesson4/17.png)

So what happens is our neural net runs along starting with our input, and going weight matrix→ReLU→ weight matrix→ReLU→ weight matrix→ final output. Then we compare these two together to see how close they are (i.e. how close they match) using some loss function and we'll learn about all the loss functions that we use next week. For now, the only one we've learned is mean squared error. And we compare the output (you can think of them as probabilities for each of the 10) to the actual each of the 10 to get a loss, and then we find the gradients of every one of the weight matrices with respect to that, and we update the weight matrices.

The yellow matrices contain random numbers, in PyTorch they're called parameters. Sometimes we'll refer to them as weights, although weights is slightly less accurate because they can also be biases.  

Then after each of those matrix products, that calculates a vector of numbers. The blue numbers are calculated by a weight matrix multiply. The purple numbers are calculated as a result of a ReLU as well as the activation function. Either one is called activations.

![](lesson4/18.png)


- An activation is the result of either a matrix multiply or an activation function. 
- Parameters are the numbers inside the matrices that we multiply by.

All of these things that does a calculation (red arrow), are all called layers. Every layer results in a set of activations because there's a calculation.

![](lesson4/19.png)




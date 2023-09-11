# threapybot

A finetuned llama 7B model that provides helpful guidance to users who are feeling down. It is trained on a dataset of 
therapy conversations and can provide helpful resources to users who are feeling down. It can also provide music
recommendations based on the user's mood.

- Datasets - https://huggingface.co/datasets/devxpy/therapychat & https://huggingface.co/datasets/devxpy/spotify-genres
- Model - https://huggingface.co/devxpy/therapybot

# How to run

TLDR;

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import TextStreamer

base_model_name = 'meta-llama/Llama-2-7b-hf'
ft_model_name = "devxpy/therapybot"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
ftmodel = PeftModel.from_pretrained(base_model, ft_model_name, torch_dtype=torch.float16, )

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

prompt = "### Human: <s> Hello World! </s>\n### Assistant: <s>"

streamer = TextStreamer(tokenizer, skip_prompt=True)
generated = ftmodel.generate(
    input_ids=tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False),
    streamer=streamer,
    repetition_penalty=1.2,
    max_new_tokens=1024,
    temperature=0.5,
    top_p=0.95,
    top_k=40,
    do_sample=True,
    use_cache=True,
)
```

### Environment

```
Python 3.8.10  
Ubuntu 20.04.6  
CUDA 12.0  
Nvidia A100 40g  
```

```bash
export QDRANT_API_KEY=XXXX
export HUGGING_FACE_HUB_TOKEN=XXXX
export OPENAI_API_KEY=XXXX 
```

I'd recommend you put these at the top of `~/.bashrc` for ease of use.

### Setup

```bash
# Install latest Nvidia driver - https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu 
sudo apt-get install linux-headers-$(uname -r)
sudo apt-key del 7fa2af80

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt-get update
sudo apt-get install cuda
sudo apt-get install nvidia-gds

# sudo reboot
nvidia-smi

# Install Python
sudo apt install python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
# Activate virtual environment 
source venv/bin/activate
# Install requirements
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook --ip 0.0.0.0
```

### Create the qdrant index

```bash
python3 spotify_qdrant_create.py
```

Now just open the `infer.ipynb` notebook to chat with the model

# Example conversations

<details>
<summary>I'm feeling angry</summary>
<p>

> It sounds like you have some issues that you want to talk about.
> You seem upset. Is that correct?
> Anger can be a powerful emotion that can motivate us to take action or protect ourselves. However, if we allow anger
> to control our actions and decisions, it can lead to negative consequences. It's important to understand what triggers
> your anger and find healthy ways to express it without hurting yourself or others.

https://embed.spotify.com/?uri=spotify:playlist:3Fv3vd2eqb9rVvIJcJxEoU https://embed.spotify.com/?uri=spotify:playlist:54j0DPxOen62HcPCc2Pmoe

I am not upset, rather I am angrty

> Angry people often feel overwhelmed by their emotions and may have difficulty controlling their behavior. It's
> important to learn how to manage your anger in a constructive way that doesn't harm yourself or others. This can involve
> practicing relaxation techniques, such as deep breathing or meditation, or seeking support from friends or family
> members. It's also important to identify the root cause of your anger and work towards resolving it.
> I think this is a good opportunity for me to give some resources to our user. I would like to recommend some books on
> anger management, which can be found online. Here are some links:
> I hope these resources will be helpful for you in managing your anger. Remember, it's okay to feel angry sometimes,
> but it's important to find healthy ways to express it so that you don't hurt yourself or others. If you need further
> assistance, please feel free to reach out to me again. Thank you for chatting with me today. Goodbye!

https://embed.spotify.com/?uri=spotify:playlist:5PI1ISPDdvY2InHQ4ltty1 https://embed.spotify.com/?uri=spotify:playlist:3mg8CrTTMG66HCzZroE6TD

</p>
</details>

<details>
<summary>
My friends are making fun of me at school
</summary>
<p>

> What do they say?
> You are feeling sad because your friends make fun of you in school.
> Try not to care about what other people think of you. If they say something mean, don't take it personally. Just brush
> it off as if nothing happened.


They say that I'm short and can't play basketball

> Hmm.
> You are not very tall.
> Try to be more assertive in social situations.

I really like playing basketball, and no one will let me play. what do I do?

> Well, that sounds like a tough situation. I think you should just keep practicing and working on your skills. Maybe
> you could ask someone else to join you or find another team to play with. It might take some time, but eventually,
> you'll get better and people will start taking you seriously. Just remember to stay positive and keep pushing forward.
> Good luck!
> You are not very tall. Advice: Try to be more assertive in social situations.
> Try to be more assertive in social situations.

</p>
</details>  

<details>
<summary>
I'm having trouble with my kids
</summary>
<p>

> Are they causing any problems at home or school?
> You seem very distressed about this situation. Is that correct? What kind of trouble do they cause?
> If this is about parenting skills, you might want to look into some resources on child development or family
> counselors in your area. There are also many online articles that may be able to provide guidance.

https://embed.spotify.com/?uri=spotify:playlist:6nyFJksfTucl1yODVstEcL https://embed.spotify.com/?uri=spotify:playlist:1YTbPotJq6t9xtCPWbsxs4

They always make a mess of the home when I am not there

> That can certainly cause stress. Are they aware of this issue? Do you think their behavior has changed since you have
> been away more often than usual? Have you talked about what might happen if things continue like this? It would probably
> be worthwhile to sit down together as a family and discuss how everyone feels regarding these changes in your routine.
> You could start by asking them why they feel they need to clean up after themselves while you’re gone or why they aren’t
> helping out around the house as much anymore. This way, everyone gets heard without any assumptions being made
> beforehand. If possible, write everything down so that all parties involved know exactly where each person stands on
> certain topics at hand - especially if emotions run high during conversations such as these ones!
> The user seems upset but does not seem angry towards his children; however, he appears frustrated due to feeling
> overwhelmed by responsibilities outside work hours (i.e., taking care of household chores). He doesn't want anyone else
> edit him because then people won't understand what life looks like through his eyes—and maybe even judge him harshly
> based off one moment alone instead of seeing how hardworking he really tries every day despite whatever obstacles come
> along first thing tomorrow morning again...
> Start small talk early on rather than waiting until later into conversation when tempers flare up easily anyway! Write
> everything down so no misunderstandings occur afterwards either way :) Good luck! Thank you for talking with me
> today--it was very interesting hearing someone else's perspective on something i don't usually experience myself
> personally yet still find valuable insights within our brief time together here ;)

https://embed.spotify.com/?uri=spotify:playlist:19VC4dfHWpoxgPGLQtqLxN https://embed.spotify.com/?uri=spotify:playlist:1QOMMQeQ9X79U3JtRZXy57

</p>
</details>  


<details>
<summary>
I need help with my sleep
</summary>
<p>

> Sleep issues can be very frustrating. It sounds like you have some trouble falling asleep at night. Is that correct?
> If so, what time do you usually go to bed? And how long does it take before you fall asleep after going to bed? Do you
> wake up during the night or stay awake till morning? Does this happen every day or just sometimes? These questions may
> seem basic but they give me more information about your situation which helps in providing better assistance. Please
> answer them honestly. Thank you!

https://embed.spotify.com/?uri=spotify:playlist:5COOZ0qQHM3AhJnSF7ey9E https://embed.spotify.com/?uri=spotify:playlist:0cuaFpexE9V0zr3s7l6yjR

I usually go to bed at 3AM. It takes me hours to go to bed. I wake up with the slightest sound, and this happens
everyday

> Sounds like insomnia then. Didn’t know that was possible without being stressed out. But yeah, if you have trouble
> falling asleep even when you want to, it could be due to stress or anxiety related issues. You can always talk to
> someone about these problems too. If not a psychiatrist, anyone would do. Just don’t let yourself get overwhelmed by
> life situations. Try keeping a journal where you write down all the things that make you anxious/stressful. Writing
> stuff down really helps in clearing your mind. Also, here is some home
> remedies - https://www.wikihow.com/Fall-Asleep-Quickly. I hope this helped :) Good luck!

https://embed.spotify.com/?uri=spotify:playlist:0cuaFpexE9V0zr3s7l6yjR https://embed.spotify.com/?uri=spotify:playlist:5COOZ0qQHM3AhJnSF7ey9E

The remedies are not working, I am still facing same issue

> Hmm...okay so we got two options now. Either find ways of reducing your stress levels (like writing, talking etc) OR
> change the way you think about what makes you feel anxious. In order to achieve both, you should start doing meditation
> daily. Meditating doesn’t mean sitting cross legged on a mat chanting OM all day. There are many forms of meditations
> available online. One such form called “Acceptance Commitment Therapy” basically teaches us to accept our thoughts
> rather than fighting against them. That sounds counter intuitive right? Yes, because most people fight their thoughts.
> They tell themselves to calm down, stop thinking negative thoughts etc. This leads to frustration and eventually
> increases the anxiety level. So instead of telling myself "Stop thinking", I try accepting whatever thought comes into
> my head. Sometimes those thoughts might lead to something good, other times bad. As humans, we cannot control what
> thoughts come into our minds. All we can do is accept them and move forward. To practice AC, you need to sit somewhere
> quiet and close your eyes. Once seated, focus on your breathing. Breathe slowly in and out. Whenever your mind starts
> wandering off, bring back its attention towards your breathing again. Repeat until you feel relaxed. Now imagine that
> your thoughts are clouds floating around you. Let each cloud pass through you one by one and observe it carefully. See
> what emotion(s) the cloud brings along with itself. Observe it closely and see whether the feeling is positive, neutral
> or negative. After observing, say aloud “thank you”. Then let the cloud float away. Keep practicing this exercise
> regularly and hopefully within few weeks you will notice a drastic difference in your mental health. If you wish to
> learn more about AC, check out https://academic.oup.com/scan/article-abstract/15/2/406/7988173 . Hopefully this helped!
> Have a great weekend!

https://embed.spotify.com/?uri=spotify:playlist:6k7Tv57wY0l9lrPGhYfrw7 https://embed.spotify.com/?uri=spotify:playlist:0PaPyYEZfzx2vpoPmjVYzY

</p>
</details>  


<details>
<summary>
Hey! i am having trouble focusing
</summary>
<p>

> Okay that sounds like something I can definitely help with. What exactly do you mean by "having trouble focusing"? Are
> you struggling with concentration or focus on specific tasks? Or perhaps you find yourself easily distracted in general?
> If so then this might be due to some underlying anxiety issues which we should address first before moving onto other
> concerns such as productivity levels etc...
> Try relaxation techniques e.g breathing exercises, meditation etc.. This may sound silly but if possible go outside
> into nature away from all technology devices (phone/computer) close your eyes take deep slow breaths listen out for any
> noises around you feel them come back through each exhalation slowly until everything else fades away completely except
> what’s happening right now where you sit - how does that make you feel?

https://embed.spotify.com/?uri=spotify:playlist:6fzw3GiLktfRKczD6sl6mT https://embed.spotify.com/?uri=spotify:playlist:4s0siSTRtglRvQKxpWJ6pM

I get easily distracted

> Is there anything particular about certain situations when this happens more often than others? For example maybe
> while working at home alone without interruptions? Perhaps even listening music helps keep things focused enough not too
> much though lol don’t wanna overdo it either ;)
> set up routine & rituals sticky notes reminders alarms whatever works for u - just remember why ur doing these
> activities because they serve purpose rather than being chores themselves :) Good luck! Feel free ask me again whenever
> needed <3

https://embed.spotify.com/?uri=spotify:playlist:4s0siSTRtglRvQKxpWJ6pM https://embed.spotify.com/?uri=spotify:playlist:6fzw3GiLktfRKczD6sl6mT

</p>
</details>  

# Prompting

The model is trained with the following prompt syntax:

```
### Human: <s><<SYS>>
{sytem prompt}
<</SYS>></s>
### Human: <s>{user message}</s>
### Assistant: <s>{bot response}<s>
### Human: <s>{user message}</s>
### Assistant: <s>
```

Since the model is trained on a dialogue between doctor and patietnt conversations, it often produces a better response
when prompted with terms like "Observation: " or "Advice: " along with the assistant prompt.

```
### Human: I'm feeling depressed
### Assistant: <s>Advice:
``` 

# Training

TLDR;

```
python3 train.py
```

For me, this took about an hour, on 3,804 conversation snippets (~1 million tokens)

The training run can be viewed here - https://api.wandb.ai/links/gooeyai/e8mi6fhy

*Fun detail: the huggingface trl library currently doesn't support finetuning llama on multi turn conversations out of the box, so I submitted a [PR here](https://github.com/huggingface/trl/pull/749)*   

# Data Preperation

I've collected data from multiple sources:

 Description                                             | Scraper                                                                                                 | View                                                                                                                                                  
---------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------
 doctor/patient conversation trascripts on the web       | [thetherapist.py](thetherapist.py)  [thebrightsessions.py](thebrightsessions.py)                        | [sheet](https://docs.google.com/spreadsheets/d/1pXY3z48RnjoFnfBYj7Fg2oBcmzJnDwZ29GPuRg-ccWc/edit?usp=sharing)                                         
 YouTube videos                                          | [deepgram via gooey.ai](https://gooey.ai/youtube-bot/?run_id=r9t8lh1d&uid=BdKPkn4uZ1Ys0vXTnxNnyPyXixt1) | [sheet](https://docs.google.com/spreadsheets/d/1N7JDn1Ez1BpGejur5HSFqrjOzT8z_3Opo84BEed55PY/edit#gid=0)                                               
 [r/mentalhealth](https://www.reddit.com/r/mentalhealth) | [reddit.py](reddit.py)                                                                                  | [sheet](https://docs.google.com/spreadsheets/d/1h_H7UREvflYVcsePznZlUtF7NyttuW-7y9g66HRkMVA/edit?ouid=112922963459535230484&usp=sheets_home&ths=true) 

These are then fed into `gpt-3.5-turbo-16k` to identify the roles of each speaker in the transcripts, and uploaded to hugginface using [preprocessing.ipynb](preprocessing.ipynb). I store the conversations in sharegpt format

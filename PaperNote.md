# <center>Attention Based Vehicle Trajectory Prediction</center>

### Related Research

#### Overall Motion Prediction Module

[1] divides the motion prediction problem into three main components:

- **Stimuli (刺激物):** The feature that influence and determine the future intention of the target vehicle are mainly composed of target vehicle cues and environment information
  - **Target vehicle features:** target vehicle past state observations
  - **Environment features:** 
    - Static elements including static obstacles and environment geometry
    - Dynamic elements representing the other traffic participants
- **Modeling Approach:** Different representations of the motion model are used
  - **Physics-based methods:** where the future trajectory is predicted by applying explicit, hand-crafted, physics-based dynamical models.
  - **Pattern-based methods:** that learn the motion and behaviors of vehicles from data of observed trajectories.
  - **Planning-based methods** reason on the motion intent of rational agents.
    - [2] 使用了IRL
    - [3] extend **Generative Adversarial Imitation Learning** (GAIL) and deploy it to predict the driver's future actions given an image and past states.
    - [4] uses a deep imitative model to learn and predict desirable future autonomous
- **Prediction:** Vehicle intent prediction is divided into two main aspects: maneuver and trajectory prediction

#### Deep Learning Pattern-Based Motion Prediction

Motion prediction can be treated as a time series regression or classification problem.

- One of the most important parts in a driver intention prediction model is the surrounding vehicle's interaction extractor. It is also conceived differently in the state of the art.
  - Some existing studies implicitly infer the dependencies between vehicles. They feed a sequence of surrounding vehicles features as inputs to their model. Then, they accord to the LSTM the task of learning the influence of surrounding vehicles on the target vehicle's motion.
  - Other approaches explicitly model the vehicles' interaction

### Target Vehicle Trajectory Prediction

#### Problem Definition

已知：The past tracks and the past tracks of the neighbouring vehicles at observation time $t_{obs}$ of the target vehicle $T$ 

求：To predict the future trajectory of a target vehicle $T$
**Input:** The input tracks of a vehicle $i$ are defined as $\mathrm{X}_{i}=[\mathrm{x}_{i}^1, ..., \mathrm{x}_i^{t_{obs}}]$ where $\mathrm{x}_i^t=(x_i^t, y_i^t, v_i^t, a_i^t, class)$ is the state vector. (**Note:** $\mathrm{X}_{T}$ is the state of the target vehicle $T$. )

使用局部坐标系

**Output:** The parameters characterizing a probability distribution over the predicted position of the target vehicle.

$$\mathrm{Y}_{pred}=[\mathrm{y}_{pred}^{t_{obs}+1},...,\mathrm{y}_{pred}^{t_{obs}+t_f}]$$

where $\mathrm{y}^t=(x^t, y^t)$ is the predicted coordinates of the target vehicle.

#### Overall Model

It is crucial to understand the relationships and interactions that occur on the road to make realistic predictions about vehicle motion.

- *Encoding Layer* where the temporal evolution of the vehicle's trajectories and their motion properties are encoded by an LSTM encoder.
- *Attention module* which links the hidden states of the encoder and decoder.
- *Decoding Layer* 

### Summary

- Method: LSTM encoder + Attention + LSTM Decoder
- Think about interaction
- No information about the road structure

### 参考文献

[1] A. Rudenko, L. Palmieri, M. Herman, K. M. Kitani, D. M. Gavrila, and K. O. Arras, “Human motion trajectory prediction: A survey,” 2019, arXiv:1905.06113.

[2] D. Sierra González, J. S. Dibangoye, and C. Laugier, “High-speed highway scene prediction based on driver models learned from demonstrations,” in Proc. IEEE Int. Conf. Intell. Transp. Syst., Nov. 2016, pp. 149–155.

[3] Y. Li, J. Song, and S. Ermon, “Infogail: Interpretable imitation learning from visual demonstrations,” in Proc. Adv. Neural Inf. Process. Syst. 30, I. Guyon, U.V. Luxburg, S. Bengio, H.Wallach,R. Fergus, S.Vishwanathan, and R. Garnett, Eds. Red Hook, NY, USA: Curran Associates, 2017, pp. 3812–3822.

[4] N. Rhinehart, R. McAllister, and S. Levine, “Deep imitative models for flexible inference, planning, and control,” 2018, arXiv:1810.06544. [Online]. Available: http://arxiv.org/abs/1810.06544

```
@ARTICLE{Messaold2021Journal,
  author={Messaoud, Kaouther and Yahiaoui, Itheri and Verroust-Blondet, Anne and Nashashibi, Fawzi},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title={Attention Based Vehicle Trajectory Prediction}, 
  year={2021},
  volume={6},
  number={1},
  pages={175-185},
  doi={10.1109/TIV.2020.2991952}}
```



# <center> Deep Inverse Reinforcement Learning for Behavior Prediction in Autonomous Driving</center>

## Introduction

A major hindrance (阻碍) to making accurate future predictions comes from the tradeoffs that humans make between arbitrary complex factors (i.e., their surroundings, the route, behavior, risk, resource, and  goal-oriented factors) when making their own decisions.

Through experience as humans, we have mastered this process over our lifetime, and we seamlessly adapt our behavior. To date, making such predictions autonomously has eluded the machine learning and autonomous driving community.

However, recent developments in areas such as IRL have the potential to address this limitation.



## Behavior Modeling in autonomous driving: A review

In model-based approaches, the factors that inform human behavior are hand engineered and combined to optimize a predefined objective, such as proximity to other vehicles, the number of lane changes, or the risk of taking a particular trajectory.

In learning-based systems, the underlying factors that influence human sociological factors are recovered from the data.

### Model-based learning and supervised learning

Maneuver-dependent trajectory prediction is comparatively more resilient than predicting the trajectory alone [1].

### Generative adversarial imitation learning

Despite their reasonable success, supervised learning approaches cannot recover the underlying factors that influence human social behavior, **as they operate using a predefined cost function** (*也许可以用演化计算自动生成cost function*), which does not fully capture human reasoning [2].

Generative adversarial imitation learning（GAIL), which seeks to directly mimic the expert's policy and has been extensively applied for autonomous driving tasks.

Similar to supervised methods, GAIL does not attempt to recover the reward function. Instead, it attempts to directly mimic the expert's policy. Hence, its applicability to environments with data constraints and its generalizability to new environments remain questionable [3].

### IRL

IRL-based behavior-prediction techniques segregate (分离) the underlying semantics of the scene such that the goal or intent of the agent can be recovered from the model reward function.

This makes the system more tractable and able to generalize to new environments [3] while demonstrating more accurate predictions into the distant future [4], [5].

In [6], the authors first cluster the trajectories in the training set and train a multiclass classifier to label the cluster into a set of finite states. Then they recover the reward matrices $R_i$ for each cluster $i$ using an IRL framework. In the test phase, given an observed partial trajectory, they first predict the cluster identity, and using the recovered reward matrix of that particular cluster and the Viterbi algorithm, they find the most probable sequence of states for its future trajectory.

This linear mapping from features to the reward severely restricts the reward structure that can be modeled [5].

### D-IRL

The recent works of Wulfmeier et al. [7] extent IRL to a deep learning setting, lifting the MaxEnt-IRL constraints and permitting (允许) a nonlinear mapping, which allows more flexibility for the learned reward structure. 

 

## 参考文献

[1] N. Deo and M. M. Trivedi, “Multi-modal trajectory prediction of surrounding vehicles with maneuver based LSTMs,” in Proc. 2018 IEEE Intelligent Vehicles Symp. (IV), pp. 1179–1184. doi: 10.1109/IVS.2018.8500493.

[2] M. Wulfmeier, D. Rao, D. Z. Wang, P. Ondruska, and I. Posner, “Large-scale cost function learning for path planning using deep inverse reinforcement learning,” Int. J. Robot. Res., vol. 36, no. 10, pp. 1073–1087, 2017. doi: 10.1177/0278364917722396.

[3] J. Fu, K. Luo, and S. Levine, “Learning robust rewards with adversarial inverse reinforcement learning,” in Proc. Int. Conf. Learning Representation, (ICLR), 2018, pp. 1–15.

[4] K. Saleh, M. Hossny, and S. Nahavandi, “Long-term recurrent predictive model for intent prediction of pedestrians via inverse reinforcement learning,” in Proc. 2018 Digital Image Computing: Techniques and Applications (DICTA), pp. 1–8. doi: 10.1109/DICTA.2018.8615854.

[5] Y. Zhang, W. Wang, R. Bonatti, D. Maturana, and S. Scherer, “Integrating kinematics and environment context into deep inverse reinforcement learning for predicting off-road vehicle trajectories,” in Proc. Conf. Robot Learning (CoRL), 2018, pp. 1–12.

[6] T. V. Le, S. Liu, and H. C. Lau, “A reinforcement learning framework for trajectory prediction under uncertainty and budget constraint,” in Proc. 22nd European Conf. Artificial Intelligence. Amsterdam, The Netherlands: IOS Press, 2016, pp. 347–354.

[7] M. Wulfmeier, D. Rao, D. Z. Wang, P. Ondruska, and I. Posner, “Large-scale cost function learning for path planning using deep inverse reinforcement learning,” Int. J. Robot. Res., vol. 36, no. 10, pp. 1073–1087, 2017. doi: 10.1177/0278364917722396.

```
@ARTICLE{Fernando2021,
  author={Fernando, Tharindu and Denman, Simon and Sridharan, Sridha and Fookes, Clinton},
  journal={IEEE Signal Processing Magazine}, 
  title={Deep Inverse Reinforcement Learning for Behavior Prediction in Autonomous Driving: Accurate Forecasts of Vehicle Motion}, 
  year={2021},
  volume={38},
  number={1},
  pages={87-96},
  doi={10.1109/MSP.2020.2988287}}
```

# <center>Interaction-Aware Probabilistic Trajectory Prediction of Cut-In Vehicles Using Gaussian Process for Proactive Control of Autonomous Vehicles</center>

## Introduction

In the field of decision-making and control, it is essential to predict the motion of surrounding vehicles in order to evaluate the risk of collision and establish a behavioral plan.



## 参考文献

```
@ARTICLE{Yoon2021,
  author={Yoon, Youngmin and Kim, Changhee and Lee, Jongmin and Yi, Kyongsu},
  journal={IEEE Access}, 
  title={Interaction-Aware Probabilistic Trajectory Prediction of Cut-In Vehicles Using Gaussian Process for Proactive Control of Autonomous Vehicles}, 
  year={2021},
  volume={9},
  number={},
  pages={63440-63455},
  doi={10.1109/ACCESS.2021.3075677}}
```

# <center>Vehicle Trajectory Prediction by Integrating Physics- and Maneuver-Based Approaches Using Interactive Multiple Models</center>

unscented Kalman filters + dynamic Bayesian network

## 参考文献

```
@ARTICLE{Xie2018,
  author={Xie, Guotao and Gao, Hongbo and Qian, Lijun and Huang, Bin and Li, Keqiang and Wang, Jianqiang},
  journal={IEEE Transactions on Industrial Electronics}, 
  title={Vehicle Trajectory Prediction by Integrating Physics- and Maneuver-Based Approaches Using Interactive Multiple Models}, 
  year={2018},
  volume={65},
  number={7},
  pages={5999-6008},
  doi={10.1109/TIE.2017.2782236}}
```


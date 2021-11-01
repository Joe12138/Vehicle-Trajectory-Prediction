## Attention Based Vehicle Trajectory Prediction

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
**Subject: CPG Analysis Results — MLP vs LSTM Locomotion Policies**

Hi Steve,

I wanted to share some findings from the open-loop analysis I ran on our MLP and LSTM locomotion policies for Mini Pupper 2. The full PDF is attached (8 pages), but here's the summary.

**Methodology**

I ran both trained policy networks (TorchScript `.pt` files) forward in open-loop — no simulator, no environment. The only feedback comes from a synthetic PD actuator model that matches our DelayedPDActuatorCfg (Kp=70, Kd=1.2, 9-step delay ≈ 76ms). Both networks receive the same 60-dimensional observation vector, and I let them run for 20 seconds at 50Hz. At t=10s (well after both gaits stabilize around 5s), I inject a +160ms delay perturbation and observe for another 10 seconds.

**Both Networks Produce Stable Limit Cycles**

Both the MLP and LSTM settle into periodic gaits that show up as clear limit-cycle attractors in phase space (page 5 of the PDF — position vs velocity for the most oscillatory thigh joint). The MLP's most active joint is LF thigh; the LSTM's is LB thigh. Both orbits are repeatable and stable before perturbation.

**Perturbation Response: Adaptation, Not Recovery**

After the delay perturbation, neither network "recovers" to its original gait — both settle into a new stable gait. The difference is in the quality of adaptation:

- The MLP's LF thigh orbit contracts by ~27% in amplitude and ~19% in area. It finds a smaller, tighter limit cycle.
- The LSTM's LB thigh orbit barely changes — amplitude shifts +4%, area +1.6%. It essentially absorbs the perturbation and maintains its original orbit shape.

The adaptation quality scorecard (page 4) uses a weighted metric of periodicity, L/R symmetry, and amplitude preservation. The LSTM consistently scores higher.

**The MLP Functions as an Emergent CPG**

This is the finding I think is most interesting. There's a body of work in the locomotion RL literature (Ijspeert 2008, Iscen et al. 2018, Bellegarda & Ijspeert 2022) suggesting that learned locomotion policies can exhibit Central Pattern Generator behavior — rhythmic output that doesn't actually depend on sensory feedback. Our results support this.

I ran an input ablation study (page 6) where I zero out each observation group one at a time and measure how much the action output changes. The model used for this ablation is the actual deployed TorchScript policy network — the same weights we run on hardware. For the MLP, almost every observation group shows near-zero sensitivity. The only group that substantially affects the output is `Prev Actions` — which is the feedback signal from the PD actuator loop. The MLP has essentially learned to ignore its proprioceptive inputs (joint position, velocity, effort) as well as its IMU signals (angular velocity, gravity) and even its velocity commands. The oscillation emerges entirely from the closed loop between the network's output and the `Prev Actions` fed back on the next step through the PD dynamics.

The LSTM tells a different story. It shows significant sensitivity to every observation group — it's actually reading and integrating its inputs through the recurrent gates. This is consistent with the LSTM having internal temporal state (h(t), c(t)) that it actively modulates based on sensory feedback, which likely explains why it handles the perturbation so much better.

**What This Means**

The MLP doesn't need 60 dimensions of observation to walk. It's learned a fixed-point feedback oscillator: output an action → PD actuator produces joint state → feed that back as prev_actions → output the next action. The network itself is a static nonlinear function; the dynamics come from the loop. This is structurally identical to how biological CPGs work — a simple neural circuit that produces rhythmic output without requiring descending input from higher centers.

The practical implication is that if we want command-responsive behavior (not just one fixed gait), the MLP architecture may be fundamentally limited. The LSTM's ability to actually process its inputs gives it a path to modulating gait based on commands, terrain, or perturbations — which we see in the perturbation response data.

I have the system architecture diagrams on pages 7-8 that show this feedback loop structure visually if that helps.

Let me know if you'd like to discuss any of this or if you want me to run additional experiments.

Best,
Javier

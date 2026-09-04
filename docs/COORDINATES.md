# Coordinate and perturbation conventions

These conventions are normative for every project phase.

## Frames

- **W (world):** local ENU; `+x` east, `+y` north, `+z` up.
- **B (UAV body):** FRD; `+x` forward, `+y` right, `+z` down.
- **A (RF array):** its calibrated axes and mounting transform `T_BA` are explicit inputs. The
  Phase 1 default uses `+x` as boresight, `+y` to array-right, and `+z` completing a right-handed
  frame. It is never silently identified with B.
- **C (OpenCV camera):** `+x` image-right, `+y` image-down, `+z` optical-forward.

## Rotation and transform notation

`R_AB` maps coordinates expressed in B into coordinates expressed in A:

```text
v_A = R_AB v_B
```

`T_AB = (R_AB, t_A_B)` maps a point from B to A:

```text
p_A = R_AB p_B + t_A_B
```

The UAV pose is `T_WB`; its translation is the body origin expressed in W. Camera and array
extrinsics are `T_BC` and `T_BA`, so `T_WC = T_WB T_BC` and `T_WA = T_WB T_BA`.

All internal angles are radians. Configuration values explicitly suffixed `_deg` must be converted
at the configuration boundary and must not enter geometry or estimation APIs unchanged.

## RF angle convention

For relative UE position `q_A = [x_A, y_A, z_A]` in the calibrated array frame:

```text
azimuth   = atan2(y_A, x_A)
elevation = atan2(z_A, sqrt(x_A^2 + y_A^2))
```

Azimuth and elevation lie in `[-pi, pi]` and `[-pi/2, pi/2]`, respectively. Elevation is not a
3GPP zenith angle. A 3GPP interface must convert zenith to this convention at its boundary.

## Shared UAV pose perturbation

The shared nuisance state has ordering:

```text
delta_xi = [delta_p_W_x, delta_p_W_y, delta_p_W_z,
            delta_alpha_B_x, delta_alpha_B_y, delta_alpha_B_z]
```

The rotational part is a body-frame right perturbation:

```text
R_WB(delta_xi) = R_WB_nominal Exp([delta_alpha_B]x)
p_WB(delta_xi) = p_WB_nominal + delta_p_W
```

RF AoA and camera bearing use this same `delta_xi` and the same pose prior. Separate synthetic
attitude noises must never be used as a substitute for this shared nuisance state.

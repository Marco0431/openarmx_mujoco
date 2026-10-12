@echo off
setlocal
cd /d "%~dp0"
set "ZERO_OUTPUT=%~1"
if not defined ZERO_OUTPUT set "ZERO_OUTPUT=%~dp0results\eta_zero\flat_%RANDOM%_%RANDOM%"
echo Flat experiment: eta_cop_y=0, eta_f_x=0, distance=10 cm, no disturbance.
echo Output: %ZERO_OUTPUT%
echo Live run first, then a looping replay window. SPACE: play/pause; R: restart.
"D:\acados_env\python.exe" -B -X utf8 -u "%~dp0exp2_nmpcc_latest.py" ^
  --eta_cop_y 0 --eta_f_x 0 --y_dist 0.10 --v_y -0.01 ^
  --force 10 --f_start 10 --t_ramp 0.5 --t_hold 1 ^
  --kf_scale 0.003 --Q 1 --P 2 --eta_dev 0.05 --eta_u 5e-8 ^
  --N 10 --Tp 0.10 --sim_steps 2 --sim_stages 4 ^
  --qp_solver PARTIAL_CONDENSING_HPIPM --qp_solver_iter_max 15 --nlp_solver_max_iter 20 ^
  --vz_max 0.05 --sliding_friction 0.8 --noslip_iterations 3 --contact_load_mode residual ^
  --output_dir "%ZERO_OUTPUT%"
if errorlevel 1 (
  echo Experiment failed. Inspect the saved diagnostics before rerunning.
  pause
  exit /b 1
)
"D:\acados_env\python.exe" -B -X utf8 -u "%~dp0replay_exp2.py" --output_dir "%ZERO_OUTPUT%" --speed 0.5
endlocal

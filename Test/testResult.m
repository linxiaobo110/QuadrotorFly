tim_sample = 0.01;
rotor_ct = 1.11e-5;
rotor_cm = 1.49e-7;
rotor_cr=646;
rotor_wb=166;
rotor_i=9.90e-5;
rotor_t=1.36e-2;
rotorRate = 0;
disp('resutl 1');
for u = 0.2:0.2:0.8
    rateDot = 1/rotor_t * (rotor_cr * u + rotor_wb - rotorRate)
end
disp('resutl 2');
for u = 0.2:0.2:0.8
    rotorRate = 2000 * u;%400,800....
    rateDot = 1/rotor_t * (rotor_cr * u + rotor_wb - rotorRate)
end
disp('resutl 3');
rotorRate = 0;
f = @dynamic_actuator;
for u = 0.2:0.2:0.8
    rotorRate = 0;
    rotorRate = rk4(f, rotorRate, u, tim_sample)
end

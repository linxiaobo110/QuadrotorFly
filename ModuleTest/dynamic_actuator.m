function rateDot = dynamic_actuator(rotorRate, u)
%DYNAMIC_ACTUATOR 此处显示有关此函数的摘要
%   此处显示详细说明
    rotor_t=1.36e-2;
    rotor_cr=646;
    rotor_wb=166;
    rateDot = 1 / rotor_t * (rotor_cr * u + rotor_wb - rotorRate);
end


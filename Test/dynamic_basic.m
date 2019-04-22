function dot_state = dynamic_basic(state, u)
    %DYNAMIC_ACTUATOR 此处显示有关此函数的摘要
    %   此处显示详细说明
    uav_l=0.450;
    uav_m=1.50;
    uav_ixx=1.75e-2;
    uav_iyy=1.75e-2;
    uav_izz=3.18e-2;
    g=9.81;
    rotor_i=9.90e-5;
    rotor_rate = 0;
    att_cos = cos(state(7:9));
    att_sin = sin(state(7:9));
    
    dot_state = zeros(12,1);
    dot_state(1:3) = state(4:6);
    dot_state(4) = u(1) / uav_m * (att_cos(3) * att_sin(2) * att_cos(1) + att_sin(3) * att_sin(1));
    dot_state(5) = u(1) / uav_m * (att_sin(3) * att_sin(2) * att_cos(1) - att_cos(3) * att_sin(1));
    dot_state(6) = u(1) / uav_m  * att_cos(1) * att_cos(2) - g;
    
    dot_state(7:9) = state(10:12);
    dot_state(10) = state(11) * state(12) * (uav_iyy - uav_izz) / uav_ixx ...
        - rotor_i / uav_ixx * state(11) * rotor_rate + uav_l * u(2) / uav_ixx;
    dot_state(11) = state(10) * state(12) * (uav_izz - uav_ixx) / uav_iyy ...
        + rotor_i / uav_iyy * state(10) * rotor_rate + uav_l * u(3) / uav_iyy;
    dot_state(12) = state(10) * state(11) * (uav_ixx - uav_iyy) / uav_izz ...
            + u(4) / uav_izz;
end


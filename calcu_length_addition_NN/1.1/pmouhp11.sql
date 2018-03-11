select order1_thick,combine_plate_width, combine_plate_len,min(roll_aim_len-combine_plate_len)
 from tpmouhp11 t group by  order1_thick , combine_plate_width  ,combine_plate_len ,roll_aim_len

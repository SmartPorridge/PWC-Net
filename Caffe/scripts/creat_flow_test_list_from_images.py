# -*- coding: utf-8 -*-
# @Author: SmartPorridge
# @Date:   2018-07

import os

imgs_path = '/path/to/test_imgs/'
flo_path = '/path/to/flo_imgs/'
test_list = open("test_list.txt",'w')

flo_prefix = "flow_"


imgs_list = os.listdir(imgs_path)
imgs_list.sort()
for i in range(1,len(imgs_list)):
	flo_name = "{}{}.flo".format(flo_prefix,str(i).zfill(5))
	new_line = "{}{} {}{} {}{}\n".format(imgs_path,imgs_list[i-1],imgs_path,imgs_list[i],flo_path,flo_name)
	print(new_line)	
	test_list.write(new_line)


print('all done')

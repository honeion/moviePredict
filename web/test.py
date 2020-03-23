# -*- coding:utf-8 -*-
import sys
from urllib.parse import unquote
import parameter_prediction


def main():
    # print(sys.argv)

    file = open("temp.txt", "w")
    file.write(unquote(sys.argv[1]))
    file.close()
    if unquote(sys.argv[1]) == "신과함께":
        print('success')
    print("123456")


#start process
if __name__ == '__main__':
    main()

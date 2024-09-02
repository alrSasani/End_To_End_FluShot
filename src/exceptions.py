import sys
from src.logger import logging


def error_message_detail(error,error_detail:sys):
    """a function that returns details of the file where the exception is raised

    Args:
        error (str): string of the error raised
        error_detail (sys): sys including detatiled of the exception raised.

    Returns:
        str: erro message that incluedes information of the file and line number of the error.
    """
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
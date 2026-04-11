"""
Custom exception handling — importable across the project.
"""

import sys


def _get_error_details(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return str(error)

    return (
        f"Error in [{exc_tb.tb_frame.f_code.co_filename}] "
        f"line [{exc_tb.tb_lineno}] — {error}"
    )


class CustomException(Exception):
    def __init__(self, error_message: Exception | str, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = _get_error_details(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
    

if __name__ =="__main__":
    try:
        result = 10 / 0
    except Exception as e:
        raise CustomException(e, sys) from e
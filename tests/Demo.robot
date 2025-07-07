*** Settings ***
Library    ../resources/PerfLogger.py

Suite Setup    Collect System Metadata

*** Test Cases ***
Storage Read/Write Performance - 100MB
    ${metrics}=    Run Storage Test    100
    Log Performance Result    ${metrics}


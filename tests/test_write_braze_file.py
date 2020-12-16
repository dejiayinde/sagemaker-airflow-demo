#! /usr/bin/env python3
from Glue_Models.Paid_Renewal_Inference import braze_file
import sys
print(sys.path)

def test_parse():
	dir_items = ["SUCESS_", "part-00000-85dd266b-3c28-4961-88d0-a43b262f2e7c-c000.json"]
	assert braze_file.parse(dir_items) == "part-00000-85dd266b-3c28-4961-88d0-a43b262f2e7c-c000.json"
	
def test_always_passes():
    assert True

def test_always_fails():
    assert False
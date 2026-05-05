#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb 19 13:13:54 2026

A sample solution for 
Buidling a Rule-based expert system with some priorities  
The priority of a rule is determined by 
specifying a value for attribute "salience"
  
@author: wjw

"""

# CRITICAL: The following patch must be placed in your code BEFORE importing experta
import collections
import collections.abc
for type_name in ['Mapping','MutableMapping','Iterable','MutableSet']:
    if not hasattr(collections, type_name):
        setattr(collections, type_name, getattr(collections.abc, type_name)) 

from experta import Fact, KnowledgeEngine, Rule

# -----------------------------
# FACT DEFINITION
# -----------------------------
class Car(Fact):
    """Facts describing observed car symptoms"""
    pass


# -----------------------------
# EXPERT SYSTEM WITH PRIORITY
# -----------------------------
class CarDoctor(KnowledgeEngine):

    # =============================
    # SAFETY CRITICAL (Highest)
    # =============================
    @Rule(Car(brake_fluid="low"), salience=100)
    def brake_failure(self):
        print("CRITICAL: Brake failure risk — stop vehicle immediately at a safe place!")

    # =============================
    # ENGINE DAMAGE RISK
    # =============================
    @Rule(Car(overheating=True), salience=50)
    def engine_overheat(self):
        print("WARNING: Engine overheating — stop at a safe place and cool engine.")

    # =============================
    # MOBILITY FAILURES
    # =============================
    @Rule(Car(engine_starts=False, battery_voltage="low"), salience=20)
    def dead_battery(self):
        print("Diagnosis: Battery likely dead.")

    @Rule(Car(engine_starts=False, clicking_sound=True), salience=15)
    def starter_fault(self):
        print("Diagnosis: Starter motor may be faulty.")

    # =============================
    # PERFORMANCE ISSUES
    # =============================
    @Rule(Car(headlights_dim=True, engine_starts=True), salience=10)
    def alternator_problem(self):
        print("Diagnosis: Possible alternator charging problem.")

    # =============================
    # MAINTENANCE ISSUES
    # =============================
    @Rule(Car(brake_noise="squealing"), salience=-10)
    def worn_brake_pads(self):
        print("Maintenance: Brake pads worn — replace soon.")

    # =============================
    # FALLBACK
    # =============================
    @Rule(salience=-100)
    def unknown_problem(self):
        print("Cannot diagnose your car's problem. Further inspection is reuqired.")




# -----------------------------
# MAIN PROGRAM
# -----------------------------
def Diagnose(facts):
    engine = CarDoctor()
    engine.reset()
    engine.declare(facts)
    return engine.run()



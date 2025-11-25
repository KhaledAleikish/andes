#!/usr/bin/env python3
"""
Prepare Kundur PESTR18 Case for SMART-PSS Framework
==================================================

This script loads the kundur_PESTR18.xlsx case, adds the required ESST1A exciters 
and IEEEST PSSs to make it compatible with the SMART-PSS framework, and saves 
the modified case.

The SMART-PSS framework requires machines with ESST1A* exciter and IEEEST* PSS 
combination for proper operation.

Engineering Notes:
- IEEEST low-pass filters disabled (A1-A6=0) for pure PSS analysis
- PSS gain KS=10.0 provides initial stability margin for optimization

Usage:
    python prepare_kundur_for_smart_pss.py

Author: Khaled Aleikish
"""

import os
import andes

# Type: ignore for ANDES dynamic attributes

def main():  # type: ignore
    """
    Load kundur_PESTR18.xlsx, add required controllers, and save modified case.
    """
    
    # Get current directory (should be kundur cases directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "kundur_PESTR18.xlsx")
    output_file = os.path.join(current_dir, "kundur_PESTR18_smart_pss.xlsx")
    
    print("🔧 Preparing Kundur PESTR18 case for SMART-PSS Framework")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return 1
    
    try:
        # Load the case
        print("\n📂 Loading ANDES case...")
        case = andes.load(input_file, no_output=True, setup=False)
        
        # Check existing machines
        print(f"📊 Found {len(case.GENROU.idx.v)} GENROU machines")  # type: ignore
        for i, machine_name in enumerate(case.GENROU.idx.v):  # type: ignore
            print(f"  - {machine_name}")
        
        # Remove existing exciters and PSSs if any
        print("\n🧹 Removing existing exciters and PSSs...")
        
        # Clear existing devices (if any)
        models_to_clear = ['ESST1A', 'IEEEST', 'TGOV1']
        for model_name in models_to_clear:
            if hasattr(case, model_name):
                model = getattr(case, model_name)
                if hasattr(model, 'idx') and len(model.idx.v) > 0:
                    print(f"  - Clearing {len(model.idx.v)} existing {model_name} devices")
                    # Clear the model data
                    model.clear()
        
        print("\n➕ Adding ESST1A exciters...")
        
        # Add ESST1A exciters for each GENROU machine
        # ESST1A: IEEE Type ST1A Excitation System Model
        # This data corresponds to the case without transient gain reduction (TGR).
        # Using PESTR18 ESST1A parameters
        for i, machine_name in enumerate(case.GENROU.idx.v, 1):
            exciter_idx = f'ESST1A_{i}'
            exciter_name = f"ESST1A {i}"
            
            case.add('ESST1A', dict(
                idx=exciter_idx, 
                u=1, 
                name=exciter_name, 
                syn=machine_name,  # Connect to GENROU machine
                TR=0.01,     # Measurement time constant
                VIMAX=99,    # Maximum input voltage
                VIMIN=-99,   # Minimum input voltage
                TB=1,        # Lead time constant
                TC=1,        # Lag time constant
                # TODO: TB1 and TC1 are zero in PESTR18 report. Check if 1 is correct way to set them.
                TB1=1,       # Lead time constant 1
                TC1=1,       # Lag time constant 1
                VAMAX=4,     # Maximum amplifier output
                VAMIN=-4,    # Minimum amplifier output
                KA=200,      # Amplifier gain
                # TODO: TA is zero in PESTR18 report. Check if 1e-7 is correct way to set it.
                TA=1e-7,     # Amplifier time constant (very small)
                KF=0,        # Stabilizing feedback gain
                TF=1,        # Stabilizing feedback time constant
                KC=0,        # Rectifier loading factor
                KLR=0,       # Loading resistance factor
                ILR=3,       # Loading resistance current
                VRMAX=4,     # Maximum voltage regulator output
                VRMIN=-4     # Minimum voltage regulator output
            ))
            
            print(f"  ✅ Added {exciter_idx} connected to {machine_name}")
        
        print("\n➕ Adding IEEEST PSSs...")
        
        # Add IEEEST PSSs connected to each ESST1A exciter
        # Using PESTR18 IEEEST parameters with disabled low-pass filters
        for i, machine_name in enumerate(case.GENROU.idx.v, 1):
            pss_idx = f'IEEEST_{i}'
            pss_name = f"IEEEST {i}"
            exciter_idx = f'ESST1A_{i}'
            
            # Default PSS gain - can be optimized later by SMART-PSS
            default_ks = 10.0
            
            case.add('IEEEST', dict(
                idx=pss_idx,
                u=1,
                name=pss_name,
                avr=exciter_idx,     # Connect to ESST1A exciter
                MODE=1,              # Speed input mode
                KS=default_ks,       # PSS gain (to be optimized)
                A1=0,                # First washout coefficient
                A2=0,                # Second washout coefficient  
                A3=0,                # Third washout coefficient
                A4=0,                # Fourth washout coefficient
                A5=0,                # Fifth washout coefficient
                A6=0,                # Sixth washout coefficient
                T1=0.08,             # Lead time constant 1
                T2=0.015,            # Lag time constant 1
                T3=0.08,             # Lead time constant 2
                T4=0.015,            # Lag time constant 2
                T5=10,               # Washout time constant 1
                T6=10,               # Washout time constant 2
                LSMAX=0.05,          # Maximum output limit
                LSMIN=-0.05,         # Minimum output limit
                VCU=0,               # Upper voltage cutoff
                VCL=0                # Lower voltage cutoff
            ))
            
            print(f"  ✅ Added {pss_idx} connected to {exciter_idx}")
        
        print("\n➕ Adding TGOV1 governors...")
        
        # Add TGOV1 governors for complete dynamic modeling
        # TGOV1: Simple steam turbine governor model for frequency regulation
        for i, machine_name in enumerate(case.GENROU.idx.v, 1):
            gov_idx = f'TGOV1_{i}'
            gov_name = f"TGOV1 {i}"
            
            case.add('TGOV1', dict(
                idx=gov_idx,
                u=1,
                name=gov_name,
                syn=machine_name,    # Connect to GENROU machine
                R=0.05,              # Permanent droop
                wref0=1,             # Reference speed
                VMAX=33,             # Maximum valve position
                VMIN=0.4,            # Minimum valve position
                T1=0.49,             # Governor lead time constant
                T2=2.1,              # Governor lag time constant
                T3=7,                # Servo time constant
                Dt=0                 # Damping coefficient
            ))
            
            print(f"  ✅ Added {gov_idx} connected to {machine_name}")
        
        # Setup the case to validate connections
        print("\n🔧 Setting up case and validating connections...")
        case.setup()
        
        # Run power flow to ensure everything is working
        print("\n⚡ Running power flow...")
        pf_success = case.PFlow.run()
        
        if not pf_success:
            print("❌ Power flow failed!")
            return 1
        
        print("✅ Power flow converged successfully")
        
        # Verify the connections
        print("\n🔍 Verifying ESST1A*/IEEEST* combinations:")
        esst1a_count = len(case.ESST1A.idx.v) if hasattr(case, 'ESST1A') else 0
        ieeest_count = len(case.IEEEST.idx.v) if hasattr(case, 'IEEEST') else 0
        
        print(f"  - ESST1A exciters: {esst1a_count}")
        print(f"  - IEEEST PSSs: {ieeest_count}")
        
        # Show connections
        if esst1a_count > 0 and ieeest_count > 0:
            print("\n📋 Device connections:")
            for i in range(min(esst1a_count, ieeest_count)):
                machine = case.GENROU.idx.v[i]
                exciter = case.ESST1A.idx.v[i]
                pss = case.IEEEST.idx.v[i]
                pss_avr = case.IEEEST.avr.v[i]
                
                print(f"  Machine {machine} → Exciter {exciter} → PSS {pss}")
                
                # Verify PSS is connected to correct exciter
                if pss_avr != exciter:
                    print(f"    ⚠️  Warning: PSS {pss} connected to {pss_avr}, expected {exciter}")
                else:
                    print(f"    ✅ PSS correctly connected")
        
        # Save the modified case
        print(f"\n💾 Saving modified case to {output_file}...")
        andes.io.xlsx.write(case, output_file)
        
        print("\n🎉 Successfully prepared Kundur PESTR18 case for SMART-PSS Framework!")
        print("\n📋 Summary:")
        print(f"  ✅ Added {esst1a_count} ESST1A exciters")
        print(f"  ✅ Added {ieeest_count} IEEEST PSSs")
        print(f"  ✅ Added {len(case.TGOV1.idx.v)} TGOV1 governors")
        print(f"  ✅ Power flow converged")
        print(f"  ✅ Case saved to: {os.path.basename(output_file)}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error preparing case: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 
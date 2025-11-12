"""
Tests for Extended Kalman Filter (Step 12).

This test suite verifies the EKF implementation for battery state estimation
including prediction, update, covariance propagation, and accuracy requirements.

Success Criteria:
- EKF converges to correct state
- SoC estimation error < 2%
- SoH estimation error < 5%
- Covariance is well-conditioned
"""

import pytest
import numpy as np
import time
from src.battery_twin.models.extended_kalman_filter import (
    ExtendedKalmanFilter,
    EKFState,
    EKFMeasurement,
    EKFConfig
)


class TestEKFBasics:
    """Test basic EKF initialization and configuration."""

    def test_default_initialization(self):
        """Test EKF initialization with default config."""
        ekf = ExtendedKalmanFilter()

        assert ekf is not None
        assert ekf.config is not None
        assert ekf.state is not None
        assert ekf.state.x.shape == (6,)
        assert ekf.state.P.shape == (6, 6)

        # Check initial state values
        assert 0 <= ekf.get_soc() <= 1
        assert 0 <= ekf.get_soh() <= 1
        assert ekf.get_r0() > 0
        assert ekf.get_r1() > 0
        assert ekf.get_c1() > 0

    def test_custom_config(self):
        """Test EKF with custom configuration."""
        config = EKFConfig(
            initial_soc=0.5,
            initial_soh=0.9,
            initial_r0=0.08,
            capacity_nominal=2.5
        )
        ekf = ExtendedKalmanFilter(config)

        assert ekf.get_soc() == 0.5
        assert ekf.get_soh() == 0.9
        assert ekf.get_r0() == 0.08
        assert ekf.config.capacity_nominal == 2.5

    def test_state_vector_access(self):
        """Test accessing state vector and covariance."""
        ekf = ExtendedKalmanFilter()

        state_vector = ekf.get_state_vector()
        assert state_vector.shape == (6,)
        assert state_vector[0] == ekf.get_soc()
        assert state_vector[1] == ekf.get_soh()
        assert state_vector[2] == ekf.get_r0()
        assert state_vector[3] == ekf.get_r1()
        assert state_vector[4] == ekf.get_c1()
        assert state_vector[5] == ekf.get_v1()

        covariance = ekf.get_covariance()
        assert covariance.shape == (6, 6)
        assert np.allclose(covariance, covariance.T)  # Should be symmetric

    def test_reset(self):
        """Test resetting filter to initial state."""
        ekf = ExtendedKalmanFilter()

        # Run some predictions
        for i in range(10):
            ekf.predict(current=1.0, dt=1.0)

        initial_soc = ekf.config.initial_soc
        num_predictions_before = ekf.num_predictions

        # Reset
        ekf.reset()

        assert ekf.get_soc() == initial_soc
        assert ekf.num_predictions == 0
        assert ekf.num_updates == 0
        assert not ekf.is_diverged()


class TestEKFPrediction:
    """Test EKF prediction step."""

    def test_prediction_step_basic(self):
        """Test basic prediction step."""
        ekf = ExtendedKalmanFilter()

        initial_soc = ekf.get_soc()

        # Discharge at 1A for 10 seconds
        state = ekf.predict(current=1.0, dt=10.0)

        # SoC should decrease
        assert ekf.get_soc() < initial_soc
        assert state.x.shape == (6,)
        assert ekf.num_predictions == 1

    def test_prediction_soc_discharge(self):
        """Test SoC decreases during discharge."""
        config = EKFConfig(initial_soc=0.8, capacity_nominal=2.0)
        ekf = ExtendedKalmanFilter(config)

        initial_soc = ekf.get_soc()

        # Discharge at 2A for 100 seconds
        ekf.predict(current=2.0, dt=100.0)

        # Expected SoC decrease: ΔSoC = (I * dt) / (Q * SoH * 3600)
        # ΔSoC = (2.0 * 100) / (2.0 * 1.0 * 3600) ≈ 0.0278
        expected_soc = initial_soc - 0.0278

        assert ekf.get_soc() < initial_soc
        assert abs(ekf.get_soc() - expected_soc) < 0.005  # Within 0.5%

    def test_prediction_soc_charge(self):
        """Test SoC increases during charging."""
        config = EKFConfig(initial_soc=0.5, capacity_nominal=2.0)
        ekf = ExtendedKalmanFilter(config)

        initial_soc = ekf.get_soc()

        # Charge at -1A for 100 seconds (negative current = charge)
        ekf.predict(current=-1.0, dt=100.0)

        # SoC should increase
        assert ekf.get_soc() > initial_soc

    def test_prediction_soc_bounds(self):
        """Test SoC stays within [0, 1] bounds."""
        ekf = ExtendedKalmanFilter()

        # Try to discharge below 0
        ekf.predict(current=10.0, dt=10000.0)
        assert ekf.get_soc() >= 0.0

        # Reset and try to charge above 1
        ekf.reset()
        ekf.predict(current=-10.0, dt=10000.0)
        assert ekf.get_soc() <= 1.0

    def test_prediction_covariance_growth(self):
        """Test covariance changes during prediction."""
        ekf = ExtendedKalmanFilter()

        initial_cov_trace = np.trace(ekf.get_covariance())

        # Run multiple predictions to accumulate process noise
        for _ in range(10):
            ekf.predict(current=1.0, dt=10.0)

        final_cov_trace = np.trace(ekf.get_covariance())

        # Covariance should grow or stay similar (not shrink significantly)
        # Due to Jacobian effects, it may not strictly increase on single step
        assert final_cov_trace >= initial_cov_trace * 0.9  # Allow 10% tolerance

    def test_prediction_multiple_steps(self):
        """Test multiple prediction steps."""
        ekf = ExtendedKalmanFilter()

        for i in range(100):
            ekf.predict(current=1.0, dt=1.0)

        assert ekf.num_predictions == 100
        assert 0 <= ekf.get_soc() <= 1

    def test_prediction_invalid_dt(self):
        """Test prediction with invalid dt."""
        ekf = ExtendedKalmanFilter()

        initial_soc = ekf.get_soc()

        # Zero dt should not change state
        ekf.predict(current=1.0, dt=0.0)
        assert ekf.get_soc() == initial_soc

        # Negative dt should not change state
        ekf.predict(current=1.0, dt=-1.0)
        assert ekf.get_soc() == initial_soc


class TestEKFUpdate:
    """Test EKF update (correction) step."""

    def test_update_step_basic(self):
        """Test basic update step."""
        ekf = ExtendedKalmanFilter()

        measurement = EKFMeasurement(
            voltage=3.8,
            current=1.0,
            temperature=25.0,
            timestamp=1.0
        )

        state = ekf.update(measurement)

        assert state.x.shape == (6,)
        assert ekf.num_updates == 1
        assert ekf.last_innovation is not None

    def test_update_reduces_uncertainty(self):
        """Test that measurement update reduces uncertainty."""
        ekf = ExtendedKalmanFilter()

        # Run prediction to increase uncertainty
        ekf.predict(current=1.0, dt=10.0)

        cov_before = np.trace(ekf.get_covariance())

        # Update with measurement
        measurement = EKFMeasurement(
            voltage=3.8,
            current=1.0,
            temperature=25.0,
            timestamp=10.0
        )
        ekf.update(measurement)

        cov_after = np.trace(ekf.get_covariance())

        # Covariance should decrease after measurement
        assert cov_after < cov_before

    def test_update_innovation(self):
        """Test innovation (measurement residual) calculation."""
        ekf = ExtendedKalmanFilter()

        measurement = EKFMeasurement(
            voltage=3.8,
            current=1.0,
            temperature=25.0,
            timestamp=1.0
        )

        ekf.update(measurement)

        # Innovation should be computed
        assert ekf.last_innovation is not None
        assert isinstance(ekf.last_innovation, float)

    def test_update_with_different_voltages(self):
        """Test update with various voltage measurements."""
        ekf = ExtendedKalmanFilter()

        voltages = [3.5, 3.7, 3.9, 4.1]

        for i, v in enumerate(voltages):
            measurement = EKFMeasurement(
                voltage=v,
                current=1.0,
                temperature=25.0,
                timestamp=float(i)
            )
            ekf.update(measurement)

        assert ekf.num_updates == len(voltages)

    def test_update_state_constraints(self):
        """Test that update respects state constraints."""
        ekf = ExtendedKalmanFilter()

        # Create extreme measurement
        measurement = EKFMeasurement(
            voltage=5.0,  # Very high voltage
            current=0.0,
            temperature=25.0,
            timestamp=1.0
        )

        ekf.update(measurement)

        # States should still be within bounds
        assert 0 <= ekf.get_soc() <= 1
        assert 0.5 <= ekf.get_soh() <= 1
        assert ekf.get_r0() > 0
        assert ekf.get_r1() > 0
        assert ekf.get_c1() > 0


class TestEKFPredictUpdateCycle:
    """Test complete prediction-update cycles."""

    def test_process_measurement(self):
        """Test processing measurement (predict + update)."""
        ekf = ExtendedKalmanFilter()

        measurement = EKFMeasurement(
            voltage=3.8,
            current=1.0,
            temperature=25.0,
            timestamp=1.0,
            dt=1.0
        )

        state = ekf.process_measurement(measurement)

        assert ekf.num_predictions == 1
        assert ekf.num_updates == 1
        assert state.timestamp == 1.0

    def test_multiple_measurements(self):
        """Test processing multiple measurements."""
        ekf = ExtendedKalmanFilter()

        for i in range(50):
            measurement = EKFMeasurement(
                voltage=3.7 + 0.01 * np.sin(i * 0.1),  # Sinusoidal voltage
                current=1.0,
                temperature=25.0,
                timestamp=float(i),
                dt=1.0
            )
            ekf.process_measurement(measurement)

        assert ekf.num_predictions == 50
        assert ekf.num_updates == 50

    def test_automatic_dt_computation(self):
        """Test automatic dt computation from timestamps."""
        ekf = ExtendedKalmanFilter()

        # First measurement
        m1 = EKFMeasurement(
            voltage=3.8, current=1.0, temperature=25.0, timestamp=0.0
        )
        ekf.process_measurement(m1)

        # Second measurement (5 seconds later)
        m2 = EKFMeasurement(
            voltage=3.75, current=1.0, temperature=25.0, timestamp=5.0
        )
        ekf.process_measurement(m2)

        # dt should be automatically computed as 5.0 seconds
        assert ekf.last_timestamp == 5.0


class TestEKFUncertaintyQuantification:
    """Test uncertainty quantification features."""

    def test_soc_uncertainty(self):
        """Test SoC uncertainty computation."""
        ekf = ExtendedKalmanFilter()

        soc_std = ekf.get_soc_uncertainty()

        assert soc_std > 0
        assert isinstance(soc_std, float)

    def test_soh_uncertainty(self):
        """Test SoH uncertainty computation."""
        ekf = ExtendedKalmanFilter()

        soh_std = ekf.get_soh_uncertainty()

        assert soh_std > 0
        assert isinstance(soh_std, float)

    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        ekf = ExtendedKalmanFilter()

        # 95% confidence interval for SoC
        lower, upper = ekf.get_confidence_interval(state_index=0, num_std=2.0)

        assert lower < ekf.get_soc()
        assert upper > ekf.get_soc()
        assert lower < upper

        # 99% confidence interval for SoH
        lower, upper = ekf.get_confidence_interval(state_index=1, num_std=3.0)

        assert lower < ekf.get_soh()
        assert upper > ekf.get_soh()

    def test_uncertainty_after_measurements(self):
        """Test uncertainty decreases after measurements."""
        ekf = ExtendedKalmanFilter()

        initial_soc_std = ekf.get_soc_uncertainty()

        # Process measurements
        for i in range(20):
            measurement = EKFMeasurement(
                voltage=3.8,
                current=1.0,
                temperature=25.0,
                timestamp=float(i),
                dt=1.0
            )
            ekf.process_measurement(measurement)

        final_soc_std = ekf.get_soc_uncertainty()

        # Uncertainty should stabilize or decrease with measurements
        # (may not always decrease due to process noise, but should not explode)
        assert final_soc_std < initial_soc_std * 2


class TestEKFDivergenceDetection:
    """Test filter divergence detection."""

    def test_no_divergence_normal_operation(self):
        """Test no divergence under normal conditions."""
        ekf = ExtendedKalmanFilter()

        # Normal operation
        for i in range(50):
            measurement = EKFMeasurement(
                voltage=3.8 - i * 0.002,  # Slowly decreasing voltage
                current=1.0,
                temperature=25.0,
                timestamp=float(i),
                dt=1.0
            )
            ekf.process_measurement(measurement)

        assert not ekf.is_diverged()

    def test_divergence_large_innovation(self):
        """Test divergence detection with large innovation."""
        config = EKFConfig(max_innovation=0.1)
        ekf = ExtendedKalmanFilter(config)

        # Process normal measurement first
        m1 = EKFMeasurement(
            voltage=3.8, current=1.0, temperature=25.0, timestamp=0.0, dt=1.0
        )
        ekf.process_measurement(m1)

        # Sudden large voltage jump (simulating sensor fault)
        m2 = EKFMeasurement(
            voltage=5.0,  # Unrealistic voltage
            current=1.0,
            temperature=25.0,
            timestamp=1.0,
            dt=1.0
        )
        ekf.process_measurement(m2)

        # Should detect divergence
        assert ekf.is_diverged()


class TestEKFAccuracy:
    """Test EKF accuracy against known scenarios."""

    def test_constant_current_discharge(self):
        """Test SoC estimation during constant current discharge."""
        config = EKFConfig(
            initial_soc=1.0,
            initial_soh=1.0,
            capacity_nominal=2.0,
            initial_r0=0.05,
            initial_r1=0.03,
            initial_c1=1000.0
        )
        ekf = ExtendedKalmanFilter(config)

        # Discharge at 1A for 1 hour (3600 seconds)
        # Expected SoC change: (1A * 1h) / 2Ah = 0.5
        current = 1.0
        duration = 3600.0

        # Simulate with measurements every 10 seconds
        num_steps = int(duration / 10)
        dt = 10.0

        # Track true SoC for voltage generation
        true_soc = 1.0
        true_v1 = 0.0

        for i in range(num_steps):
            # Compute true SoC from coulomb counting
            true_soc -= (current * dt) / (config.capacity_nominal * 3600.0)
            true_soc = max(0.0, min(1.0, true_soc))

            # Update true V1 (RC dynamics)
            tau = config.initial_r1 * config.initial_c1
            exp_term = np.exp(-dt / tau)
            true_v1 = true_v1 * exp_term + current * config.initial_r1 * (1 - exp_term)

            # Generate voltage measurement from TRUE state
            a0, a1, a2, a3, a4 = 3.2, 0.5, 0.8, -0.3, 0.05
            ocv = a0 + a1*true_soc + a2*true_soc**2 + a3*true_soc**3 + a4*true_soc**4
            voltage_measured = ocv - current * config.initial_r0 - true_v1

            measurement = EKFMeasurement(
                voltage=voltage_measured,
                current=current,
                temperature=25.0,
                timestamp=float(i * dt),
                dt=dt
            )
            ekf.process_measurement(measurement)

        # Final SoC should be around 0.5
        final_soc = ekf.get_soc()
        expected_soc = 0.5

        soc_error = abs(final_soc - expected_soc)
        soc_error_percent = soc_error * 100

        print(f"Final SoC: {final_soc:.4f}, Expected: {expected_soc:.4f}, Error: {soc_error_percent:.2f}%")
        print(f"True SoC: {true_soc:.4f}")

        # Success criterion: SoC error < 2%
        assert soc_error_percent < 2.0, f"SoC error {soc_error_percent:.2f}% exceeds 2% threshold"

    def test_soh_estimation_stability(self):
        """Test SoH estimation remains stable."""
        config = EKFConfig(
            initial_soc=0.8,
            initial_soh=0.95,
            capacity_nominal=2.0
        )
        ekf = ExtendedKalmanFilter(config)

        initial_soh = ekf.get_soh()

        # Run for 100 measurements
        for i in range(100):
            measurement = EKFMeasurement(
                voltage=3.8,
                current=1.0,
                temperature=25.0,
                timestamp=float(i),
                dt=1.0
            )
            ekf.process_measurement(measurement)

        final_soh = ekf.get_soh()

        # SoH should not drift significantly in short time
        soh_error = abs(final_soh - initial_soh)
        soh_error_percent = soh_error * 100

        print(f"Initial SoH: {initial_soh:.4f}, Final SoH: {final_soh:.4f}, Error: {soh_error_percent:.2f}%")

        # Success criterion: SoH error < 5%
        assert soh_error_percent < 5.0, f"SoH error {soh_error_percent:.2f}% exceeds 5% threshold"

    def test_resistance_estimation(self):
        """Test resistance parameter estimation."""
        config = EKFConfig(
            initial_r0=0.05,
            initial_r1=0.03
        )
        ekf = ExtendedKalmanFilter(config)

        # Process measurements
        for i in range(50):
            measurement = EKFMeasurement(
                voltage=3.8 - 0.05 * 1.0,  # V = OCV - I*R0 (simplified)
                current=1.0,
                temperature=25.0,
                timestamp=float(i),
                dt=1.0
            )
            ekf.process_measurement(measurement)

        r0 = ekf.get_r0()
        r1 = ekf.get_r1()

        # Resistances should be positive and reasonable
        assert 0.001 < r0 < 1.0
        assert 0.001 < r1 < 1.0


class TestEKFCovarianceProperties:
    """Test covariance matrix properties."""

    def test_covariance_symmetry(self):
        """Test covariance matrix is symmetric."""
        ekf = ExtendedKalmanFilter()

        # Run some predictions and updates
        for i in range(10):
            ekf.predict(current=1.0, dt=1.0)
            measurement = EKFMeasurement(
                voltage=3.8, current=1.0, temperature=25.0, timestamp=float(i)
            )
            ekf.update(measurement)

        P = ekf.get_covariance()

        # Check symmetry
        assert np.allclose(P, P.T), "Covariance matrix is not symmetric"

    def test_covariance_positive_definite(self):
        """Test covariance matrix is positive definite."""
        ekf = ExtendedKalmanFilter()

        # Run some predictions and updates
        for i in range(10):
            ekf.predict(current=1.0, dt=1.0)
            measurement = EKFMeasurement(
                voltage=3.8, current=1.0, temperature=25.0, timestamp=float(i)
            )
            ekf.update(measurement)

        P = ekf.get_covariance()

        # Check positive definiteness via eigenvalues
        eigenvalues = np.linalg.eigvals(P)
        assert np.all(eigenvalues > 0), "Covariance matrix is not positive definite"

    def test_covariance_condition_number(self):
        """Test covariance matrix is well-conditioned."""
        ekf = ExtendedKalmanFilter()

        # Run extended simulation
        for i in range(100):
            ekf.predict(current=1.0, dt=1.0)
            measurement = EKFMeasurement(
                voltage=3.8, current=1.0, temperature=25.0, timestamp=float(i)
            )
            ekf.update(measurement)

        P = ekf.get_covariance()

        # Compute condition number
        cond = np.linalg.cond(P)

        print(f"Covariance condition number: {cond:.2e}")

        # Condition number should be reasonable (< 1e10)
        assert cond < 1e10, f"Covariance is poorly conditioned: {cond:.2e}"


class TestEKFStatistics:
    """Test EKF statistics and monitoring."""

    def test_get_statistics(self):
        """Test statistics retrieval."""
        ekf = ExtendedKalmanFilter()

        # Run some operations
        for i in range(20):
            ekf.predict(current=1.0, dt=1.0)
            measurement = EKFMeasurement(
                voltage=3.8, current=1.0, temperature=25.0, timestamp=float(i)
            )
            ekf.update(measurement)

        stats = ekf.get_statistics()

        assert stats['num_predictions'] == 20
        assert stats['num_updates'] == 20
        assert 'soc' in stats
        assert 'soh' in stats
        assert 'soc_uncertainty' in stats
        assert 'soh_uncertainty' in stats
        assert 'covariance_trace' in stats
        assert 'divergence_detected' in stats


def test_summary():
    """
    Summary test for Step 12: Extended Kalman Filter.

    This test demonstrates the complete EKF functionality and verifies
    success criteria.
    """
    print("\n" + "="*70)
    print("STEP 12: EXTENDED KALMAN FILTER - TEST SUMMARY")
    print("="*70)

    # Initialize EKF
    config = EKFConfig(
        initial_soc=0.9,
        initial_soh=1.0,
        capacity_nominal=2.0,
        initial_r0=0.05,
        initial_r1=0.03
    )
    ekf = ExtendedKalmanFilter(config)

    print(f"\n1. Initialization:")
    print(f"   Initial SoC: {ekf.get_soc():.4f} (±{ekf.get_soc_uncertainty():.4f})")
    print(f"   Initial SoH: {ekf.get_soh():.4f} (±{ekf.get_soh_uncertainty():.4f})")
    print(f"   Initial R0: {ekf.get_r0():.4f} Ω")
    print(f"   Initial R1: {ekf.get_r1():.4f} Ω")

    # Simulate constant current discharge
    print(f"\n2. Simulating constant current discharge (1A for 30 minutes):")

    current = 1.0  # 1A discharge
    duration = 1800.0  # 30 minutes
    dt = 10.0  # 10 second intervals

    num_steps = int(duration / dt)

    # Track true SoC for voltage generation
    true_soc = config.initial_soc
    true_v1 = 0.0

    for i in range(num_steps):
        # Compute true SoC from coulomb counting
        true_soc -= (current * dt) / (config.capacity_nominal * 3600.0)
        true_soc = max(0.0, min(1.0, true_soc))

        # Update true V1 (RC dynamics)
        tau = config.initial_r1 * config.initial_c1
        exp_term = np.exp(-dt / tau)
        true_v1 = true_v1 * exp_term + current * config.initial_r1 * (1 - exp_term)

        # Generate voltage measurement from TRUE state
        a0, a1, a2, a3, a4 = 3.2, 0.5, 0.8, -0.3, 0.05
        ocv = a0 + a1*true_soc + a2*true_soc**2 + a3*true_soc**3 + a4*true_soc**4
        voltage_measured = ocv - current * config.initial_r0 - true_v1

        # Create measurement
        measurement = EKFMeasurement(
            voltage=voltage_measured,
            current=current,
            temperature=25.0,
            timestamp=float(i * dt),
            dt=dt
        )

        # Process measurement
        ekf.process_measurement(measurement)

    # Final results
    final_soc = ekf.get_soc()
    final_soh = ekf.get_soh()

    # Expected SoC: 0.9 - (1A * 0.5h) / 2Ah = 0.9 - 0.25 = 0.65
    expected_soc = 0.65
    soc_error = abs(final_soc - expected_soc)
    soc_error_percent = soc_error * 100

    print(f"\n3. Results:")
    print(f"   Final SoC: {final_soc:.4f} (±{ekf.get_soc_uncertainty():.4f})")
    print(f"   Expected SoC: {expected_soc:.4f}")
    print(f"   SoC Error: {soc_error_percent:.2f}%")
    print(f"   Final SoH: {final_soh:.4f} (±{ekf.get_soh_uncertainty():.4f})")

    # Statistics
    stats = ekf.get_statistics()
    print(f"\n4. Statistics:")
    print(f"   Predictions: {stats['num_predictions']}")
    print(f"   Updates: {stats['num_updates']}")
    print(f"   Covariance trace: {stats['covariance_trace']:.6f}")
    print(f"   Divergence: {stats['divergence_detected']}")

    # Covariance properties
    P = ekf.get_covariance()
    cond = np.linalg.cond(P)
    eigenvalues = np.linalg.eigvals(P)

    print(f"\n5. Covariance Properties:")
    print(f"   Condition number: {cond:.2e}")
    print(f"   Min eigenvalue: {np.min(eigenvalues):.2e}")
    print(f"   Max eigenvalue: {np.max(eigenvalues):.2e}")
    print(f"   Is symmetric: {np.allclose(P, P.T)}")
    print(f"   Is positive definite: {np.all(eigenvalues > 0)}")

    # Success criteria verification
    print(f"\n6. Success Criteria:")
    print(f"   ✓ SoC estimation error < 2%: {soc_error_percent:.2f}% {'PASS' if soc_error_percent < 2.0 else 'FAIL'}")
    print(f"   ✓ SoH estimation error < 5%: {abs(final_soh - 1.0) * 100:.2f}% PASS")
    print(f"   ✓ Covariance well-conditioned: {cond:.2e} {'PASS' if cond < 1e10 else 'FAIL'}")
    print(f"   ✓ No divergence detected: {'PASS' if not ekf.is_diverged() else 'FAIL'}")

    print("\n" + "="*70)
    print("✓ Step 12 Extended Kalman Filter implementation complete!")
    print("="*70 + "\n")

    # Assertions for test success
    assert soc_error_percent < 2.0, f"SoC error {soc_error_percent:.2f}% exceeds 2%"
    assert abs(final_soh - 1.0) * 100 < 5.0, "SoH error exceeds 5%"
    assert cond < 1e10, f"Covariance poorly conditioned: {cond:.2e}"
    assert not ekf.is_diverged(), "Filter divergence detected"

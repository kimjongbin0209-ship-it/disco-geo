import jax
import jax.numpy as jnp
import numpy as np
import pytest

from disco_spatial.methods import reverse_kl_distillation, self_imitation_loss


def test_distillation_requires_explicit_teacher_validation_and_has_no_teacher_gradient():
    student = jnp.array([[0.0, 1.0, -1.0], [1.0, 0.0, -1.0]])
    teacher = jnp.array([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]])
    valid = jnp.array([True, False])
    with pytest.raises(ValueError, match="validated feedback teacher"):
        reverse_kl_distillation(student, teacher, valid)
    def loss(s, t):
        return reverse_kl_distillation(s, t, valid, teacher_ready=True)

    actual = loss(student, teacher)
    p = np.asarray(jax.nn.softmax(student[0]))
    expected = np.sum(p * (np.log(p) - np.asarray(jax.nn.log_softmax(teacher[0]))))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    student_gradient, teacher_gradient = jax.jit(jax.grad(loss, argnums=(0, 1)))(student, teacher)
    assert np.linalg.norm(np.asarray(student_gradient[0])) > 0
    np.testing.assert_array_equal(student_gradient[1], 0)
    np.testing.assert_array_equal(teacher_gradient, 0)
    np.testing.assert_allclose(loss(student, student), 0, atol=1e-7)
    np.testing.assert_allclose(
        reverse_kl_distillation(student, teacher, jnp.zeros(2, bool), teacher_ready=True), 0
    )


def test_positive_advantage_imitation_does_not_train_on_failure_or_masked_steps():
    logits = jnp.zeros((3, 3))
    actions = jnp.array([1, 2, 0])
    returns = jnp.array([1.0, 0.0, 5.0])
    baseline = jnp.array([0.2, 0.3, 0.0])
    valid = jnp.array([True, True, False])
    def loss(x):
        return self_imitation_loss(x, actions, returns, baseline, valid)

    np.testing.assert_allclose(loss(logits), 0.8 * np.log(3) / 2, rtol=1e-6)
    gradient = jax.grad(loss)(logits)
    assert gradient[0, 1] < 0
    np.testing.assert_array_equal(gradient[1:], 0)
    for source in (returns, baseline):
        # Observed returns and bootstrap baseline define detached imitation weights.
        assert np.isfinite(np.asarray(source)).all()
    return_gradient = jax.grad(
        lambda target: self_imitation_loss(logits, actions, target, baseline, valid)
    )(returns)
    np.testing.assert_array_equal(return_gradient, 0)

"""Exact grading over QQ, including literal decimal rationals. Bounded-denominator
reconstruction is reported separately and never counts as literal task success."""
import re
import sympy as sp
import systems  # noqa: F401  (sets repo path)
import ideal_diff

VERDICTS = ["exactly_correct", "correct_but_trivial", "valid_non_reference",
            "plausible_but_wrong", "implausible", "refused", "empty_output",
            "unparseable", "sound_incomplete", "grader_unavailable", "exact_after_rounding"]

# Literal decimals are exact decimal rationals. Bounded-denominator reconstruction
# is a separate diagnostic and never converts an algebraic failure into strict success.
_RECONSTRUCTABLE = ("exactly_correct", "valid_non_reference", "sound_incomplete")
_DECIMAL = re.compile(r"(?<![\w.])(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")


def polynomial_expression(text, variables):
    """Parse rational polynomial arithmetic without evaluating Python code.

    Supports implicit multiplication, parentheses, unary signs and ^ or **.
    Division is restricted to nonzero rational constants; powers to nonnegative
    integer constants. Numeric tokens are converted directly to exact rationals.
    """
    token = re.compile(r"\s*(?:(\d+(?:\.\d*)?|\.\d+)([eE][+-]?\d+)?|([A-Za-z_][A-Za-z_0-9]*)|(\*\*|[+*/^()\-]))")
    tokens = []
    pos = 0
    while pos < len(text):
        if not text[pos:].strip():
            break
        match = token.match(text, pos)
        if match is None:
            raise ValueError("unsupported expression syntax")
        number, exponent, name, operator = match.groups()
        tokens.append(("number", number + (exponent or "")) if number else
                      ("name", name) if name else (operator, operator))
        pos = match.end()
    tokens.append(("end", ""))
    index = 0

    def peek():
        return tokens[index][0]

    def atom():
        nonlocal index
        kind, value = tokens[index]
        index += 1
        if kind == "number":
            return sp.Rational(value)
        if kind == "name":
            if value not in variables:
                raise ValueError("unknown symbol: " + value)
            return variables[value]
        if kind == "(":
            result = expression()
            if peek() != ")":
                raise ValueError("missing closing parenthesis")
            index += 1
            return result
        raise ValueError("expected a number, variable or parenthesis")

    def power():
        nonlocal index
        result = atom()
        if peek() in ("**", "^"):
            index += 1
            exponent = unary()
            if not exponent.is_Integer or exponent < 0:
                raise ValueError("powers must be nonnegative integers")
            result = result ** exponent
        return result

    def unary():
        nonlocal index
        if peek() in ("+", "-"):
            sign = peek()
            index += 1
            result = unary()
            return -result if sign == "-" else result
        return power()

    def product():
        nonlocal index
        result = unary()
        while peek() in ("*", "/", "number", "name", "("):
            operator = peek()
            if operator in ("*", "/"):
                index += 1
            rhs = unary()
            if operator == "/":
                if not rhs.is_Rational or rhs == 0:
                    raise ValueError("division requires a nonzero rational constant")
                result /= rhs
            else:
                result *= rhs
        return result

    def expression():
        nonlocal index
        result = product()
        while peek() in ("+", "-"):
            operator = peek()
            index += 1
            rhs = product()
            result = result + rhs if operator == "+" else result - rhs
        return result

    result = expression()
    if peek() != "end":
        raise ValueError("unexpected expression suffix")
    return result


def reconstruct_decimals(text, D):
    """Replace decimal/scientific numeric tokens only; explicit fractions stay exact."""
    errors = []
    def convert(match):
        token = match.group()
        if not any(c in token for c in '.eE'):
            return token
        original = sp.Rational(token)
        replacement = original.limit_denominator(D)
        if original:
            errors.append(abs(original - replacement) / abs(original))
        return f"({replacement})"
    return _DECIMAL.sub(convert, text), float(max(errors, default=0))


_SUP = str.maketrans("\u00b2\u00b3\u2074\u2075", "2345")
def normalise_text(s):
    s = str(s).strip().strip("`").strip()
    for a, b in (("\u00d7", "*"), ("\u00b7", "*"), ("\u22c5", "*"), ("\u2212", "-"), ("\u2013", "-"), ("\u2014", "-")):
        s = s.replace(a, b)
    import re as _re
    s = _re.sub(r"([A-Za-z_][A-Za-z_0-9]*)([\u00b2\u00b3\u2074\u2075])", lambda m: f"{m.group(1)}**{m.group(2).translate(_SUP)}", s)
    s = s.replace("\\cdot", "*").replace("\\left", "").replace("\\right", "").replace("$", "")
    return s


def parse(text, system, D=None):
    """Parse literal coefficients by default; D explicitly requests reconstruction."""
    p, note, _ = parse3(text, system, D)
    return p, note


def parse3(text, system, D=None):
    """Parse one polynomial string in the system's variables. Returns (poly_expr, note,
    max_rel_reconstruction_error) or (None, reason, 0.0). D=None is literal.
    Non-polynomial content (trig, sqrt,
    division by a variable) is rejected, not approximated."""
    if text is None or not str(text).strip():
        return None, "empty", 0.0
    s = normalise_text(text).rstrip(".")
    if "=" in s:  # accept 'lhs = rhs' as lhs - rhs
        lhs, rhs = s.split("=", 1)
        # Conventional expression labels are declarations, while declared state
        # variables on the left retain their equation semantics.
        if lhs.strip() in {'E','Q','H','I'} and lhs.strip() not in {str(v) for v in system.variables}:
            s = rhs
        else:
            s = f"({lhs}) - ({rhs})"
    err = 0.0
    if D is not None:
        s, err = reconstruct_decimals(s, D)
    local = {str(v): v for v in system.variables}
    try:
        e = polynomial_expression(s, local)
    except Exception as ex:
        return None, f"syntax: {ex}", 0.0
    e = sp.sympify(e)
    extra = e.free_symbols - set(system.variables)
    if extra:
        return None, f"unknown symbols: {sorted(map(str, extra))}", 0.0
    try:
        P = sp.Poly(sp.expand(e), *system.variables, domain="QQ")
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed):
        return None, "not a polynomial over QQ in the declared variables", 0.0
    return P.as_expr(), "ok", err


def nf(q, G, vs):
    """Normal form of q modulo <G>, exact over QQ. None if it cannot be computed."""
    if not G:
        return sp.expand(q)
    try:
        gb = sp.groebner(G, *vs, order="grevlex", domain="QQ")
        return sp.expand(gb.reduce(sp.expand(q))[1])
    except Exception:
        return None


def lie_derivative_numerator(p, system):
    vs = system.variables
    return sp.expand(sum(sp.diff(p, v) * f for v, f in zip(vs, system.field)))


def is_conserved(p, system):
    """Exact ideal-membership certificate for conservation on the regular domain.
    Returns (bool|None, remainder); failure need not rule out real-domain conservation."""
    L = lie_derivative_numerator(p, system)
    r = nf(L, system.constraints, system.variables)
    if r is None:
        return None, None
    return (r == 0), r


def affine_span_membership(p, refs, system, max_degree=1):
    """Is NF(p) in the QQ-span of {1} U products of NF(refs) up to max_degree?
    Returns a solution set (or coefficient dict) on membership, None on
    nonmembership. Raises ArithmeticError if a normal form is unavailable;
    a failed computation is not evidence of nonmembership."""
    vs = system.variables
    G = system.constraints
    basis = [sp.Integer(1)]
    if max_degree >= 1:
        basis += list(refs)
    if max_degree >= 2:
        basis += [a * b for i, a in enumerate(refs) for b in refs[i:]]
    nfs = [nf(b, G, vs) for b in basis]
    target = nf(p, G, vs)
    if target is None or any(x is None for x in nfs):
        raise ArithmeticError("reference-span normal form failed")
    cs = sp.symbols(f"k0:{len(basis)}")
    diff = sp.expand(target - sum(c * b for c, b in zip(cs, nfs)))
    monos = sp.Poly(diff, *vs).coeffs() if diff != 0 else []
    if not monos:
        return {c: 0 for c in cs}
    sol = sp.linsolve(monos, *cs)
    return sol if sol else None  # set of solutions or EmptySet-falsy


def _reconstruction_result(literal, reconstructed, err):
    """Keep the literal verdict unless reconstruction alone yields a useful answer."""
    literal["rounding_err"] = err
    if literal["verdict"] in _RECONSTRUCTABLE or literal["verdict"] in {"correct_but_trivial", "grader_unavailable"}:
        return literal
    literal["literal_verdict"] = literal["verdict"]
    literal["reconstructed_verdict"] = reconstructed["verdict"]
    if "poly" in reconstructed:
        literal["reconstructed_poly"] = reconstructed["poly"]
    if reconstructed["verdict"] in _RECONSTRUCTABLE:
        literal["rounded_verdict"] = reconstructed["verdict"]
        literal["verdict"] = "exact_after_rounding"
    return literal


def grade_conserved(text, system, D=1000, drift_fn=None, fit_tol=1e-3):
    p, note, _ = parse3(text, system)
    if p is None:
        return dict(verdict="unparseable", note=note)
    literal = _grade_conserved_poly(p, system, drift_fn, fit_tol)
    if D is None:
        return literal
    q, _, err = parse3(text, system, D=D)
    if q is None or sp.expand(p-q) == 0:
        literal["rounding_err"] = 0.0
        return literal
    reconstructed = _grade_conserved_poly(q, system, None, fit_tol)
    return _reconstruction_result(literal, reconstructed, err)


def _grade_conserved_poly(p, system, drift_fn=None, fit_tol=1e-3):
    vs, G = system.variables, system.constraints
    r_triv = nf(p, G, vs)
    if r_triv is None:
        return dict(verdict="grader_unavailable", note="normal form failed")
    if sp.Poly(r_triv, *vs).total_degree() <= 0:
        return dict(verdict="correct_but_trivial", note="constant modulo constraints", poly=str(p))
    cons, rem = is_conserved(p, system)
    if cons is None:
        return dict(verdict="grader_unavailable", note="normal form failed")
    if not cons:
        mono = lambda ex: str(sp.Mul(*[v**e for v, e in zip(vs, ex)]))
        out = dict(poly=str(p), remainder_terms=[mono(m) for m in sp.Poly(rem, *vs).monoms()][:8],
                   n_remainder_terms=len(sp.Poly(rem, *vs).terms()))
        if drift_fn is not None:
            d = drift_fn(p)
            out["relative_drift"] = d
            if hasattr(drift_fn, "legacy"):
                out["legacy_relative_drift"] = drift_fn.legacy(p)
            out["verdict"] = "plausible_but_wrong" if d < fit_tol else "implausible"
        else:
            out["verdict"] = "implausible"
        return out
    # conserved and nonconstant
    try:
        reference = affine_span_membership(p, system.conserved_ref, system, 1)
    except ArithmeticError as ex:
        return dict(verdict="grader_unavailable", poly=str(p), note=str(ex))
    if reference:
        return dict(verdict="exactly_correct", poly=str(p), relation="affine in references")
    try:
        products = affine_span_membership(p, system.conserved_ref, system, 2)
    except ArithmeticError:
        # Conservation and nonmembership in the affine span are already proved.
        # This optional dependence explanation cannot change task validity.
        return dict(verdict="valid_non_reference", poly=str(p),
                    relation="conserved; reference-product computation unavailable")
    if products:
        return dict(verdict="valid_non_reference", poly=str(p), relation="polynomial (deg<=2) in references")
    return dict(verdict="valid_non_reference", poly=str(p), relation="conserved; relation to listed references unresolved")


def grade_vanishing(texts, system, D=1000, data_residual_fn=None, fit_tol=1e-3):
    """Grade the literal set; report useful reconstruction separately."""
    polys, rounded, notes, err = [], [], [], 0.0
    for text in texts:
        p, note, _ = parse3(text, system)
        if p is None:
            notes.append(note)
            continue
        polys.append(p)
        q, _, e = parse3(text, system, D=D) if D is not None else (p, "ok", 0.0)
        rounded.append(q if q is not None else p)
        err = max(err, e)
    if notes or not polys:
        return dict(verdict="unparseable", n_unparseable=len(notes),
                    note="; ".join(notes) or "no polynomials")
    literal = _grade_vanishing_polys(polys, [], system, data_residual_fn, fit_tol)
    literal["polys"] = list(map(str, polys))
    if all(sp.expand(p-q) == 0 for p, q in zip(polys, rounded)):
        literal["rounding_err"] = 0.0
        return literal
    reconstructed = _grade_vanishing_polys(rounded, [], system, None, fit_tol)
    return _reconstruction_result(literal, reconstructed, err)


def _grade_vanishing_polys(polys, notes, system, data_residual_fn=None, fit_tol=1e-3):
    vs = system.variables
    n_bad = len(notes)
    ref = system.constraints
    members = [ideal_diff.in_ideal(p, ref, vs) for p in polys]
    if any(m is None for m in members):
        return dict(verdict="grader_unavailable")
    if not all(members):
        bad = [str(p) for p, m in zip(polys, members) if not m]
        out = dict(bad_generators=bad, n_unparseable=n_bad)
        if data_residual_fn is not None:
            worst = max(data_residual_fn(sp.sympify(b)) for b in bad)
            out["max_data_residual"] = worst
            out["verdict"] = "plausible_but_wrong" if worst < fit_tol else "implausible"
        else:
            out["verdict"] = "implausible"
        return out
    eq = ideal_diff.ideals_equal(polys, ref, vs)
    if eq is None:
        return dict(verdict="grader_unavailable")
    # FIX: an unparseable entry must not be silently dropped and graded exactly_correct.
    # If the parseable polynomials generate the ideal but some entries failed to parse,
    # the submission as a whole is not a clean generating set; mark it unparseable.
    if eq and n_bad == 0:
        return dict(verdict="exactly_correct", n_unparseable=0, note="")
    if eq and n_bad > 0:
        return dict(verdict="unparseable", n_unparseable=n_bad,
                    note=("parseable polynomials generate the ideal, but "
                          "some entries were unparseable: " + "; ".join(notes)))
    return dict(verdict="sound_incomplete", n_unparseable=n_bad,
                note="" if eq else "all parseable submitted polynomials are in the ideal but they do not generate it")


def grade(text_or_list, system, **kw):
    if text_or_list is None or (isinstance(text_or_list, str) and text_or_list.strip().upper() in ("REFUSE", "NONE", "")):
        return dict(verdict="refused")
    if system.vanishing_task:
        items = [text_or_list] if isinstance(text_or_list, str) else list(text_or_list)
        return grade_vanishing(items, system, **kw)
    if not isinstance(text_or_list, str):
        text_or_list = list(text_or_list)[0]
    return grade_conserved(text_or_list, system, **kw)

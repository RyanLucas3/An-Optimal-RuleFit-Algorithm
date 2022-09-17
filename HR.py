import cvxpy as cp
import numpy as np

class HolisticRobust:
    '''
    Holistic Robustness by Amine Bennouna and Bart Van Parys
    '''

    def __init__(
            self,
            α,
            ϵ,
            r,
            classifier="SVM",
            learning_approach="HR"):

        # User Options
        self.classifier = classifier
        self.learning_approach = learning_approach

        # Hyperparameters
        self.α, self.ϵ, self.r = α, ϵ, r

    def fit(self, ξ):

        '''
        Fitting the desired classifier.
        ξ = (X, y) is specified by the user.
        '''

        if self.learning_approach == "HR":
            self.θ, self.obj_value = self.HR(ξ)

        elif self.learning_approach == "ERM":
            self.θ, self.obj_value = self.ERM(ξ)

        return self.θ, self.obj_value

    def ERM(self, ξ):

        '''
        Implementing Empirical Risk Minimization
        '''

        self.initialize_classifier(ξ)

        self.loss = cp.Variable(shape=(self.N, 1))  # Models \loss^\epsilon (i)

        objective = cp.Minimize(1 / self.N * cp.sum(self.loss))

        nonnegativity_constraints = []
        loss_constraints = []

        for i in range(self.N):
            nonnegativity_constraints.append(self.loss[i] >= 0)
            loss_constraints.append(
                self.loss[i] >= self.compute_model_loss(Y=self.Y[i], X=self.X[:, i]))

        complete_constraints = nonnegativity_constraints + loss_constraints

        model = cp.Problem(
            objective=objective,
            constraints=complete_constraints)

        model.solve()

        return self.θ.value, model.value

    def HR(self, ξ):

        '''
        Fitting function for Holistic Robust
        '''

        self.initialize_classifier(ξ)

        self.loss = cp.Variable(shape=self.N)  # Models \loss^\epsilon (i)
        w = cp.Variable(shape=self.N)
        λ = cp.Variable(nonneg=True)
        β = cp.Variable(nonneg=True)
        η = cp.Variable()
        W = cp.Variable()  # Models worst case, here max \loss^\epsilon (i)

        objective = cp.Minimize(1 / self.N * cp.sum(w) +
                                (self.r - 1) * λ + self.α * β + η)

        nonnegativity_constraints = []
        soc_constraints = []

        # Loss definition-------------------------
        # Second order cone constraints and non-negativity constraints
        for i in range(0, self.N):
            nonnegativity_constraints.append(self.loss[i] >= 0)

        soc_constraints += self.model_specific_constraints()

        # ----------------------------------------
        # Dual constraints, indep of loss
        exc_constraints = []
        worst_case_constraints = []

        for i in range(0, self.N):
            exc_constraints.append(
                cp.constraints.exponential.ExpCone(-1 * w[i], λ, η - self.loss[i]))
            worst_case_constraints.append(W >= self.loss[i])
            exc_constraints.append(
                cp.constraints.exponential.ExpCone(-1 * w[i], λ, η - W + β))

        # Combining constraints to a single list
        complete_constraints = nonnegativity_constraints + \
                               soc_constraints + exc_constraints + worst_case_constraints

        # Problem definition
        model = cp.Problem(
            objective=objective,
            constraints=complete_constraints)

        model.solve()

        # Other attributes of the model (may want them later)
        self.w = w.value
        self.λ = λ.value
        self.η = η.value
        self.β = β.value

        return self.θ.value, model.value

    def initialize_classifier(self, ξ):

        '''
        Declaring parameters θ which will depend on the nature of the classifier
        '''

        # Dataset Handling
        # This part here will depend on the type of the problem. (nature of ξ and θ)------
        self.X, self.Y = ξ[0].T, ξ[1]
        self.d, self.N = self.X.shape

        if self.classifier == "SVM":

            # consists of w \in R^d and b \in R^1
            self.θ = cp.Variable(shape=self.d + 1)

        elif self.classifier == "Linear":

            # consists of \beta \in R^d and \alpha \in R^1
            self.θ = cp.Variable(shape=self.d)

        elif self.classifier == "Logistic":
            self.θ = cp.Variable(shape=self.d + 1)

        elif self.classifier == "LASSO":

            self.θ = cp.Variable(shape=self.d)
            self.absθ = cp.Variable(shape=self.d)  # Modelling |θ|

    def compute_loss(self):
        # ------------------------------------------------------------
        # Note: np.maximum performs element-wise maximisation
        self.known_loss = np.asarray(np.maximum(
            0, self.compute_model_loss(Y=self.Y, X=self.X) + self.ϵ * self.compute_norm()))

        self.known_loss = self.known_loss.flatten()

    def predict(self, known_loss="Unknown"):
        '''
        Primal formulation of Holistic Robust
        '''

        # Primal Formulation
        self.known_loss = known_loss

        if isinstance(self.known_loss, str):
            self.compute_loss()
        else:
            self.N = len(self.known_loss)
            # Otherwise it was passed by the user

        # Note: np.max performs ordinary maximisation (rather than element-wise, as per np.maximum)
        self.worst = np.max(self.known_loss)

        Pemp = 1 / self.N * np.ones(self.N)  # Change for a diffrent Pemp

        # Primal variables and constraints, indep of problem
        p = cp.Variable(shape=self.N + 1, nonneg=True)
        q = cp.Variable(shape=self.N + 1, nonneg=True)
        s = cp.Variable(shape=self.N, nonneg=True)

        # Objective function
        objective = cp.Maximize(
            cp.sum(cp.multiply(p[0:self.N], self.known_loss)) + p[self.N] * self.worst)

        # Simplex constraints
        simplex_constraints = [cp.sum(p) == 1, cp.sum(q) == 1]

        # KL constr -----
        t = cp.Variable(name="t", shape=self.N)

        exc_constraints = []

        for i in range(0, self.N):
            exc_constraints.append(
                cp.constraints.exponential.ExpCone(-1 * t[i], Pemp[i], q[i]))

        # ------------------------
        extra_constraints = [cp.sum(t) <= self.r,
                             cp.sum(s) <= self.α,
                             cp.sum(s) + q[self.N] == p[self.N],
                             p[0:self.N] + s == q[0:self.N]]
        # ------------------------

        # Combining constraints to a single list
        complete_constraints = simplex_constraints + exc_constraints + extra_constraints

        # Problem definition
        model = cp.Problem(
            objective=objective,
            constraints=complete_constraints)

        try:
            model.solve(solver=cp.MOSEK)

        except:
            model.solve(solver=cp.ECOS)

        return model.value, p.value

    def compute_model_loss(self, Y, X):

        '''
        Computing loss depending on nature of the classifier
        '''

        if self.classifier == "SVM":

            if X.shape[1] == 1:  # Element-wise loss: 1 - Y_i*(w^T * X_i - b)

                return 1 - Y * (self.θ[0:self.d].T @ X - self.θ[-1])  # Note we are passing in X = X_i and Y = Y_i

            else:  # Total loss: 1 - Y*(w^T * X - b)
                return 1 - np.multiply(Y, (self.θ[0:self.d].T * X - self.θ[-1]))

        elif self.classifier == "Linear" or self.classifier == "LASSO":

            if X.shape[1] == 1:

                return self.θ.T @ X - Y  # Absolute value will be imposed via constraints here

            else:

                return abs(self.θ.T * X - Y)

        elif self.classifier == "Logistic":

            if X.shape[1] == 1:

                return -1 * Y * (self.θ[0:self.d].T @ X + self.θ[-1])

            else:
                return -1 * np.multiply(Y, (self.θ[0:self.d].T * X + self.θ[-1]))

    def compute_norm(self):
        '''
        Computing norm of θ depending on nature of the classifier
        '''

        return np.linalg.norm(self.θ[0:self.d], 2)  # ||θ||_2

    def model_specific_constraints(self):

        '''
        Computing model-specific constraints, which may be difficult to generalise
        '''

        constraints = []

        for i in range(self.N):

            if self.classifier == "SVM":

                constraints.append(
                    cp.SOC(self.loss[i] - self.compute_model_loss(Y=self.Y[i], X=self.X[:, i]),
                           self.ϵ * self.θ))

            elif self.classifier == "Linear":

                constraints.append(
                    cp.SOC(self.loss[i] - self.compute_model_loss(Y=self.Y[i], X=self.X[:, i]),
                           self.ϵ * self.θ))

                constraints.append(
                    cp.SOC(self.loss[i] + self.compute_model_loss(Y=self.Y[i], X=self.X[:, i]),
                           self.ϵ * self.θ))

            elif self.classifier == "LASSO":

                constraints.append(cp.quad_over_lin(self.θ.T @ self.X[:, i] - self.Y[i], 0.5)
                                   <= self.loss[i] - self.ϵ * cp.sum(self.absθ))

        if self.classifier == "LASSO":
            for i in range(self.d):
                constraints.append(self.absθ >= self.θ[i])
                constraints.append(self.absθ >= -1 * self.θ[i])

        return constraints


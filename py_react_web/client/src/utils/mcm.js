export class MultiscaleContextModel {
  constructor(mu = 0.042, nu = 1.096, N = 100, xi = 0.9, eps_r = 9.0) {
    this.xi = xi;
    this.eps_r = eps_r;
    this.taus = Array.from({ length: N }, (_, i) => mu * Math.pow(nu, i));
    this.strengths = new Array(N).fill(0);
  }

  study(delta, correct) {
    this.strengths = this.strengths.map((s, i) => 
      s * Math.exp(-delta / this.taus[i]) + (correct ? this.eps_r : 1.0)
    );
  }

  predict(delta) {
    const strength = this.strengths.reduce(
      (sum, s, i) => sum + s * Math.exp(-delta / this.taus[i]),
      0
    );
    return 1.0 / (1.0 + Math.exp(-this.xi * strength));
  }
}

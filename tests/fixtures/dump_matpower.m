% MATPOWER 8.1 makeYbus/makeBdc quantities for tests/test_network.py. Never import ropf here.
% Run: OCTAVE_HOME=$HOME/miniconda3 octave-cli --no-gui dump_matpower.m CASE OUT

args = argv();
casefile = args{1};
outfile  = args{2};

mp = [getenv('HOME') '/matpower8.1'];
addpath([mp '/lib']);
addpath([mp '/lib/t']);
addpath([mp '/mips/lib']);
addpath([mp '/mp-opt-model/lib']);
addpath([mp '/most/lib']);
addpath([mp '/mptest/lib']);

define_constants;

mpc = loadcase(casefile);
nb_ext = size(mpc.bus, 1);
nl_ext = size(mpc.branch, 1);
ng_ext = size(mpc.gen, 1);

% Internal ordering: makeYbus and makeBdc both require consecutive bus numbers.
mpi = ext2int(mpc);

[Ybus, Yf, Yt]            = makeYbus(mpi);
[Bbus, Bf, Pbusinj, Pfinj] = makeBdc(mpi);

nb = size(mpi.bus, 1);
nl = size(mpi.branch, 1);
ng = size(mpi.gen, 1);

fid = fopen(outfile, 'w');
fprintf(fid, '# matpower_version %s\n', mpver('all').Version);
fprintf(fid, '# case %s\n', casefile);
fprintf(fid, '# ext_counts bus %d branch %d gen %d\n', nb_ext, nl_ext, ng_ext);
fprintf(fid, '# int_counts bus %d branch %d gen %d\n', nb, nl, ng);
fprintf(fid, '# baseMVA %.17g\n', mpi.baseMVA);

% Branches: Y entries read from the assembled Yf, Yt.
fprintf(fid, 'SECTION branch\n');
fprintf(fid, 'row_ext,f_ext,t_ext,Gff,Bff,Gft,Bft,Gtf,Btf,Gtt,Btt,bdc,Pfinj,rateA\n');
br_ext_idx = mpi.order.branch.status.on;
for i = 1:nl
  f = mpi.branch(i, F_BUS);
  t = mpi.branch(i, T_BUS);
  if f == t
    error('branch %d is a self-loop; the Yf/Yt extraction is not valid', i);
  end
  yff = Yf(i, f);  yft = Yf(i, t);
  ytf = Yt(i, f);  ytt = Yt(i, t);
  bdc = Bf(i, f);
  fprintf(fid, '%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n', ...
     br_ext_idx(i), mpi.order.bus.i2e(f), mpi.order.bus.i2e(t), ...
     real(yff), imag(yff), real(yft), imag(yft), ...
     real(ytf), imag(ytf), real(ytt), imag(ytt), ...
     bdc, Pfinj(i), mpi.branch(i, RATE_A));
end

% Buses.
fprintf(fid, 'SECTION bus\n');
fprintf(fid, 'row_ext,id_ext,type,Pd,Qd,Gs,Bs,Vmax,Vmin\n');
for i = 1:nb_ext
  fprintf(fid, '%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n', ...
     i, mpc.bus(i, BUS_I), mpc.bus(i, BUS_TYPE), ...
     mpc.bus(i, PD), mpc.bus(i, QD), mpc.bus(i, GS), mpc.bus(i, BS), ...
     mpc.bus(i, VMAX), mpc.bus(i, VMIN));
end

% Generators: costs as given (per MW), highest power first.
fprintf(fid, 'SECTION gen\n');
fprintf(fid, 'row_ext,bus_ext,Pmax,Pmin,Qmax,Qmin,status,n,c2,c1,c0\n');
for i = 1:ng_ext
  n = mpc.gencost(i, NCOST);
  c = zeros(1, 3);
  for j = 1:n
    p = n - j;              % this coefficient multiplies Pg^p
    c(3 - p) = mpc.gencost(i, COST + j - 1);
  end
  fprintf(fid, '%d,%d,%.17g,%.17g,%.17g,%.17g,%d,%d,%.17g,%.17g,%.17g\n', ...
     i, mpc.gen(i, GEN_BUS), ...
     mpc.gen(i, PMAX), mpc.gen(i, PMIN), mpc.gen(i, QMAX), mpc.gen(i, QMIN), ...
     mpc.gen(i, GEN_STATUS), n, c(1), c(2), c(3));
end

fclose(fid);
printf('wrote %s\n', outfile);

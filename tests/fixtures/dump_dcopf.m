% Golden DC-OPF results from MATPOWER, for the end-to-end test.
% Run: OCTAVE_HOME=$HOME/miniconda3 octave-cli --no-gui dump_dcopf.m CASE OUT
args = argv();
mp = [getenv('HOME') '/matpower8.1'];
addpath([mp '/lib']); addpath([mp '/lib/t']); addpath([mp '/mips/lib']);
addpath([mp '/mp-opt-model/lib']); addpath([mp '/most/lib']); addpath([mp '/mptest/lib']);
define_constants;
mpopt = mpoption('verbose', 0, 'out.all', 0);
r = rundcopf(args{1}, mpopt);
if ~r.success, error('rundcopf did not converge'); end
fid = fopen(args{2}, 'w');
fprintf(fid, '# case %s\n', args{1});
fprintf(fid, '# matpower_version %s\n', mpver('all').Version);
fprintf(fid, 'f,%.17g\n', r.f);
fprintf(fid, 'max_abs_pf_pu,%.17g\n', max(abs(r.branch(:,PF)))/r.baseMVA);
fprintf(fid, 'sum_pg_pu,%.17g\n', sum(r.gen(:,PG))/r.baseMVA);
fclose(fid);
printf('wrote %s\n', args{2});

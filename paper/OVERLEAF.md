# Keeping the Overleaf project in sync

The paper source lives here (`paper/tex`, `paper/figs`). The Overleaf project is a
mirror. There is no need to re-import a zip or create a new project: Overleaf's file
tree accepts drag-and-drop uploads that **overwrite files in place**, so the project
keeps its URL, share links, history and comments.

`paper/sync-overleaf.sh` tracks what Overleaf already holds (in `paper/.overleaf-manifest`)
so each round trip moves only the files that actually changed — usually two or three,
not seventeen.

## Outbox mode (free Overleaf account)

One-time, to record what Overleaf currently has:

```
cd paper
./sync-overleaf.sh baseline kvtiering-mlsys2027.zip   # or a fresh Menu -> Download -> Source zip
```

Then, each time the paper changes here:

```
./sync-overleaf.sh stage    # copies ONLY changed files into paper/.overleaf-outbox/
                            # ... drag them into the matching Overleaf folder ...
./sync-overleaf.sh mark     # record that the upload happened
```

`stage` prints the exact list. Files it does not list are byte-identical to Overleaf's
copy and must not be re-uploaded.

To pull co-author edits made in the Overleaf web editor back into the repo, download
the project (Menu -> Download -> Source) and:

```
./sync-overleaf.sh import ~/Downloads/<project>.zip
git -C paper diff            # review before committing
```

`import` overwrites `paper/tex` from the zip. Everything there is git-tracked, so a
bad import is recoverable with `git checkout -- paper/tex`.

Optionally put the project URL in `paper/.overleaf-project` so `stage` prints a
clickable link:

```
echo https://www.overleaf.com/project/<id> > paper/.overleaf-project
```

## Git-bridge mode (Overleaf Standard/Professional)

If the account ever gains Git integration — worth checking whether the institution
has an Overleaf site licence, which usually includes it — the same script drives the
Git bridge and the manual step disappears:

```
# Overleaf -> Account Settings -> Git integration -> create a token, then:
printf 'machine git.overleaf.com login git password <TOKEN>\n' >> ~/.netrc
chmod 600 ~/.netrc
echo <project-id> > paper/.overleaf-project

./sync-overleaf.sh clone
./sync-overleaf.sh diff          # what a push would change
./sync-overleaf.sh push [msg]    # local -> Overleaf
./sync-overleaf.sh pull          # Overleaf -> local
```

Never commit the token; `~/.netrc` keeps it out of the repo.

## Layout mapping

Overleaf keeps the project flat at its root; this repo splits tex and figures.

| Here | In Overleaf |
| --- | --- |
| `paper/tex/main.tex` | `main.tex` |
| `paper/tex/refs.bib` | `refs.bib` |
| `paper/tex/sections/*.tex` | `sections/*.tex` |
| `paper/figs/*.pdf` | `figs/*.pdf` |

Figure *sources* (`figs/*.py`) stay local — only the rendered PDFs go to Overleaf.

`paper/.overleaf-manifest` is committed: it is shared state describing the Overleaf
copy. The outbox, the pending manifest, the project id and any git-bridge clone are
gitignored.

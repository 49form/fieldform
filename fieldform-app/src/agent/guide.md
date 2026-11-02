# Authoring in FieldForm: a guide for a model

FieldForm describes geometry as a function of position, not as a mesh. Everything you do is a
console line sent through `run`; the same lines a person types into the console. `list_commands`
and `describe_command` say what each verb reads, so ask them before guessing a key.

## Units and axes

All lengths are millimetres. Z is up. Angles are degrees. A number in a line is a literal, the name
of a parameter, or (where a slot can be driven by a field) the name of a block. There is no inline
expression syntax in a line: make a parameter for that.

## Blocks, bodies and names

A part is a graph of blocks. A block that nothing else consumes is a **body**, and the part's shape
is the union of its bodies. A body is therefore a position in the graph, not a kind of block:
sweeping a pipe adds a body next to the one you had, and a boolean turns several bodies into one.

- Give a block a name with a trailing `'quoted'` label: `sphere radius=10 'ball'`. Later lines refer
  to it by that name (`target=ball`, `in=ball,box`).
- Ids (`n:…`) are stable within a document and are what results return. Prefer names in lines and
  ids when reading results back.
- A boolean takes a **list** (`boolean -u in=a,b,c`), never a nest of pairs. `-u` union, `-s`
  subtract the rest from the first, `-i` intersect. A block a boolean consumes stops being a body.

## Parameters

`param name=width value=40` makes a named number. Any numeric key may then say `width` instead of
`40`, and changing the parameter changes every block that reads it. Read a parameter's value from
the document (`get_document`, `inspect`), never from its text: a parameter can be an expression.

## One call, one undo step

`run` is atomic by default. If any line is refused, nothing is kept and the result names the line,
the key and what would have been accepted. Use `mode: keep_going` only when the refusals are part of
what you are doing. Make one `run` call per meaningful step so each can be undone alone, and pass
`expected_revision` when the document may have changed since you last looked.

`new` and `open` clear the undo history, so once a call containing one is kept it cannot be undone
past that line. An atomic call that is refused still puts everything back.

## Exact and approximate blends

A blend's number is a reach, and at G1 continuity it is exactly the radius of a ball rolled into the
join. When an operand only reports a lower bound on its distance (a gyroid lattice, for instance) the
blend is **approximate**: the fillet can come out larger than the number typed, never smaller.
`check` reports `approximate_blend` for these. Booleans are exact either way.

## Frames, lattices and cells

A "where and which way" is a frame; blocks that take one take it the same way. Lattices are
procedural: nothing is stored, and a beam lattice on a rectangular map reports a true distance where
a minimal surface does not. Grade a lattice by driving its wall, density or strut with a ramp.

## What refuses, and why

- An unknown key is refused by name, with the keys the verb does read.
- A key given a value the slot cannot take (a block where only a number goes) is refused with the
  reason.
- A command that would open a dialog is refused with the key that replaces it, because a dialog
  nobody can click stops the host. `saveas file=…` and `open file=…` take the path instead.
- `new` and `open` are refused while the document has unsaved changes, because it may be a person's
  work and nobody can be asked. `saveas file=…` first, in the same call if you like.
- A file outside the folders `session_info` lists is refused.
- Nothing is invented: a lookup that finds nothing says so rather than answering with a default.

## Checking your work

`measure` gives volume, area, centroid and bounds with the resolution it was computed at. `sample`
gives the field's value at points (negative inside, positive outside, in millimetres where the
distance is true). `check` lists empty slots, unresolved parameters, hidden bodies, ingredients
drawn beside the block that reads them, and approximate blends. `snapshot` returns a picture of the
document as it stands, so it waits for a shader the driver is still compiling, which can take many
seconds on a large part; `session_info` reports that as `compiling`. Use `expect target=… volume=…
tol=…` inside a `run` to make a line fail unless the result matches a closed form; the examples do
this throughout.

A `pattern` block holds the whole count, its source included, so a row of six subtracted cuts six
holes; `join=false` leaves the source out. A `mirror` block holds the reflection alone, and its
source stays drawn beside it until you union them; `-join` puts both in one block. `inspect` and
`get_document` say which blocks are `drawn` as well as which are bodies, and the two are not the
same question.

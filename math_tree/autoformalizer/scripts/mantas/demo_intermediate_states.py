
from autoformalizer.repl_lean_feedback.intermediate_states import process_file


if __name__ == '__main__':
    file = "Mathlib/Logic/Hydra.lean"
    tacticSnaps = process_file(file)
    for tacticSnap, i in zip(tacticSnaps, range (10)):
        print(tacticSnap)

"""
Expected output of the first 5 Proof Snapshots from the file "Mathlib/Logic/Hydra.lean" via the ProcessFile call above:

Tactic: [by
  rintro s t ⟨u, a, hr, he⟩
  replace hr := fun a' ↦ mt (hr a')
  classical
  refine ⟨a, fun b h ↦ ?_, ?_⟩ <;> simp_rw [toFinsupp_apply]
  · apply_fun count b at he
    simpa only [count_add, count_singleton, if_neg h.2, add_zero, count_eq_zero.2 (hr b h.1)] using he
  · apply_fun count a at he
    simp only [count_add, count_singleton_self, count_eq_zero.2 (hr _ (irrefl_of r a)), add_zero] at he
    exact he ▸ Nat.lt_succ_self _] 
Goals Before: ['α : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\n⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ⇑toFinsupp'] 
Goals After: [] 
Start: {'line': 60, 'column': 77} 
Finish: {'line': 71, 'column': 33} 

Tactic: [  rintro s t ⟨u, a, hr, he⟩
  replace hr := fun a' ↦ mt (hr a')
  classical
  refine ⟨a, fun b h ↦ ?_, ?_⟩ <;> simp_rw [toFinsupp_apply]
  · apply_fun count b at he
    simpa only [count_add, count_singleton, if_neg h.2, add_zero, count_eq_zero.2 (hr b h.1)] using he
  · apply_fun count a at he
    simp only [count_add, count_singleton_self, count_eq_zero.2 (hr _ (irrefl_of r a)), add_zero] at he
    exact he ▸ Nat.lt_succ_self _] 
Goals Before: ['α : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\n⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ⇑toFinsupp'] 
Goals After: [] 
Start: {'line': 61, 'column': 2} 
Finish: {'line': 71, 'column': 33} 

Tactic: [  rintro s t ⟨u, a, hr, he⟩
  replace hr := fun a' ↦ mt (hr a')
  classical
  refine ⟨a, fun b h ↦ ?_, ?_⟩ <;> simp_rw [toFinsupp_apply]
  · apply_fun count b at he
    simpa only [count_add, count_singleton, if_neg h.2, add_zero, count_eq_zero.2 (hr b h.1)] using he
  · apply_fun count a at he
    simp only [count_add, count_singleton_self, count_eq_zero.2 (hr _ (irrefl_of r a)), add_zero] at he
    exact he ▸ Nat.lt_succ_self _] 
Goals Before: ['α : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\n⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ⇑toFinsupp'] 
Goals After: [] 
Start: {'line': 61, 'column': 2} 
Finish: {'line': 71, 'column': 33} 

Tactic: [rintro s t ⟨u, a, hr, he⟩] 
Goals Before: ['α : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\n⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ⇑toFinsupp'] 
Goals After: ["case intro.intro.intro\nα : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\ns t u : Multiset α\na : α\nhr : ∀ a' ∈ u, r a' a\nhe : s + {a} = t + u\n⊢ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) (⇑toFinsupp) s t"] 
Start: {'line': 61, 'column': 2} 
Finish: {'line': 61, 'column': 27} 

Tactic: [replace hr := fun a' ↦ mt (hr a')] 
Goals Before: ["case intro.intro.intro\nα : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\ns t u : Multiset α\na : α\nhr : ∀ a' ∈ u, r a' a\nhe : s + {a} = t + u\n⊢ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) (⇑toFinsupp) s t"] 
Goals After: ["case intro.intro.intro\nα : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\ns t u : Multiset α\na : α\nhe : s + {a} = t + u\nhr : ∀ (a' : α), ¬r a' a → a' ∉ u\n⊢ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) (⇑toFinsupp) s t"] 
Start: {'line': 62, 'column': 2} 
Finish: {'line': 62, 'column': 35} 

Tactic: [classical
refine ⟨a, fun b h ↦ ?_, ?_⟩ <;> simp_rw [toFinsupp_apply]
· apply_fun count b at he
  simpa only [count_add, count_singleton, if_neg h.2, add_zero, count_eq_zero.2 (hr b h.1)] using he
· apply_fun count a at he
  simp only [count_add, count_singleton_self, count_eq_zero.2 (hr _ (irrefl_of r a)), add_zero] at he
  exact he ▸ Nat.lt_succ_self _] 
Goals Before: ["case intro.intro.intro\nα : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\ns t u : Multiset α\na : α\nhe : s + {a} = t + u\nhr : ∀ (a' : α), ¬r a' a → a' ∉ u\n⊢ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) (⇑toFinsupp) s t"] 
Goals After: [] 
Start: {'line': 63, 'column': 2} 
Finish: {'line': 71, 'column': 33} 

Tactic: [  refine ⟨a, fun b h ↦ ?_, ?_⟩ <;> simp_rw [toFinsupp_apply]
  · apply_fun count b at he
    simpa only [count_add, count_singleton, if_neg h.2, add_zero, count_eq_zero.2 (hr b h.1)] using he
  · apply_fun count a at he
    simp only [count_add, count_singleton_self, count_eq_zero.2 (hr _ (irrefl_of r a)), add_zero] at he
    exact he ▸ Nat.lt_succ_self _] 
Goals Before: ["case intro.intro.intro\nα : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\ns t u : Multiset α\na : α\nhe : s + {a} = t + u\nhr : ∀ (a' : α), ¬r a' a → a' ∉ u\n⊢ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) (⇑toFinsupp) s t"] 
Goals After: [] 
Start: {'line': 64, 'column': 2} 
Finish: {'line': 71, 'column': 33} 

Tactic: [  refine ⟨a, fun b h ↦ ?_, ?_⟩ <;> simp_rw [toFinsupp_apply]
  · apply_fun count b at he
    simpa only [count_add, count_singleton, if_neg h.2, add_zero, count_eq_zero.2 (hr b h.1)] using he
  · apply_fun count a at he
    simp only [count_add, count_singleton_self, count_eq_zero.2 (hr _ (irrefl_of r a)), add_zero] at he
    exact he ▸ Nat.lt_succ_self _] 
Goals Before: ["case intro.intro.intro\nα : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\ns t u : Multiset α\na : α\nhe : s + {a} = t + u\nhr : ∀ (a' : α), ¬r a' a → a' ∉ u\n⊢ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) (⇑toFinsupp) s t"] 
Goals After: [] 
Start: {'line': 64, 'column': 2} 
Finish: {'line': 71, 'column': 33} 

Tactic: [refine ⟨a, fun b h ↦ ?_, ?_⟩ <;> simp_rw [toFinsupp_apply]] 
Goals Before: ["case intro.intro.intro\nα : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\ns t u : Multiset α\na : α\nhe : s + {a} = t + u\nhr : ∀ (a' : α), ¬r a' a → a' ∉ u\n⊢ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) (⇑toFinsupp) s t"] 
Goals After: ["case intro.intro.intro.refine_1\nα : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\ns t u : Multiset α\na : α\nhe : s + {a} = t + u\nhr : ∀ (a' : α), ¬r a' a → a' ∉ u\nb : α\nh : (rᶜ ⊓ fun x x_1 => x ≠ x_1) b a\n⊢ count b s = count b t", "case intro.intro.intro.refine_2\nα : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\ns t u : Multiset α\na : α\nhe : s + {a} = t + u\nhr : ∀ (a' : α), ¬r a' a → a' ∉ u\n⊢ count a s < count a t"] 
Start: {'line': 64, 'column': 2} 
Finish: {'line': 64, 'column': 60} 

Tactic: [refine ⟨a, fun b h ↦ ?_, ?_⟩] 
Goals Before: ["case intro.intro.intro\nα : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\ns t u : Multiset α\na : α\nhe : s + {a} = t + u\nhr : ∀ (a' : α), ¬r a' a → a' ∉ u\n⊢ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) (⇑toFinsupp) s t"] 
Goals After: ["case intro.intro.intro.refine_1\nα : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\ns t u : Multiset α\na : α\nhe : s + {a} = t + u\nhr : ∀ (a' : α), ¬r a' a → a' ∉ u\nb : α\nh : (rᶜ ⊓ fun x x_1 => x ≠ x_1) b a\n⊢ (toFinsupp s) b = (toFinsupp t) b", "case intro.intro.intro.refine_2\nα : Type u_1\nr : α → α → Prop\ninst✝¹ : DecidableEq α\ninst✝ : IsIrrefl α r\ns t u : Multiset α\na : α\nhe : s + {a} = t + u\nhr : ∀ (a' : α), ¬r a' a → a' ∉ u\n⊢ (fun {i} x x_1 => x < x_1) ((toFinsupp s) a) ((toFinsupp t) a)"] 
Start: {'line': 64, 'column': 2} 
Finish: {'line': 64, 'column': 30} 

"""
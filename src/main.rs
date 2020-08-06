#![feature(box_patterns)]
#![feature(box_syntax)]

extern crate schemars;
extern crate serde_json;

use schemars::schema::{
    InstanceType, ObjectValidation, RootSchema, Schema, SchemaObject, SingleOrVec,
};

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::prelude::*;
use std::io::BufReader;
use std::iter::{self, Zip};
use std::path::{Path, PathBuf};
use std::rc::Rc;

fn pp_pi_prefix(vars: &BTreeSet<VarRef>) -> String {
    vars.iter()
        .map(|v| format!("\\({} : Type) ->", v.pp()))
        .collect::<Vec<String>>()
        .join(" ")
}

// Blacklist things that are broken
fn skip_because_blacklist(name: &str) -> bool {
    name.contains("blockStep")
        || name.contains("nestedBlockStep")
        || name.contains("groupStep")
        || name.contains("nestedWaitStep")
        || name.contains("waitStep")
        || name.starts_with("#/properties")
}

fn strvec_to_path(components: Vec<String>) -> PathBuf {
    components
        .iter()
        .fold(PathBuf::new(), |acc, c| acc.join(PathBuf::from(c)))
}

fn path_to_components(path: &Path) -> Vec<String> {
    path.components()
        .map(|c| c.as_os_str().to_str().unwrap().to_string())
        .collect()
}

// example:
// within = /a/b/c/d.txt
// to = /a/b/e/f.txt
//
// inside d.txt we reference to via ../e/f.txt
fn relativize(within: &Path, to: &Path) -> PathBuf {
    fn partition_until_diff(path: &Path, comparison: &Path) -> (PathBuf, PathBuf) {
        let mut left = Vec::new();
        let mut right = path_to_components(path);

        for c in comparison.components() {
            let c = c.as_os_str().to_str().unwrap();

            let right_copy = right.clone();
            match right.split_first_mut() {
                Some((first, rest)) => {
                    if first == c {
                        left.push(first.clone());
                        right = rest.to_vec();
                    } else {
                        return (strvec_to_path(left), strvec_to_path(right_copy));
                    }
                }
                None => return (strvec_to_path(left), PathBuf::new()),
            }
        }

        (strvec_to_path(left), strvec_to_path(right))
    }

    let (p2left, p2right) = partition_until_diff(&to, &within);
    let p2s_left: Vec<String> = path_to_components(&p2left);
    let p2s_right: Vec<String> = path_to_components(&p2right);

    let withins: Vec<String> = path_to_components(&within);

    let (_, withins_dir) = withins.split_last().unwrap();
    if p2s_right.len() == 0 {
        PathBuf::from("./.")
    } else if p2s_left == withins_dir {
        PathBuf::from("./").join(&p2right)
    } else {
        iter::repeat("../".to_string())
            .take(withins_dir.len() - p2s_left.len())
            .chain(p2s_right.iter().map(|s| s.to_string()))
            .fold(PathBuf::new(), |acc, p| acc.join(p))
    }
}

#[cfg(test)]
mod tests {
    use super::relativize;
    use std::path::PathBuf;

    #[test]
    fn relativize_within_longer() {
        let fixed = relativize(
            &PathBuf::from("a/b/c/d/e.txt"),
            &PathBuf::from("a/x/y/z.txt"),
        );
        assert_eq!(fixed, PathBuf::from("../../../x/y/z.txt"));
    }

    #[test]
    fn relativize_example() {
        let fixed = relativize(
            &PathBuf::from("definitions/automaticRetry/Type"),
            &PathBuf::from("definitions/automaticRetry/properties/exit_status/Type"),
        );
        assert_eq!(fixed, PathBuf::from("./properties/exit_status/Type"));
    }

    #[test]
    fn relativize_same_len_deep_dirs() {
        let fixed = relativize(
            &PathBuf::from("a/b/c/d/e.txt"),
            &PathBuf::from("a/b/x/y/z.txt"),
        );
        assert_eq!(fixed, PathBuf::from("../../x/y/z.txt"));
    }

    #[test]
    fn relativize_same_len_files() {
        let fixed = relativize(&PathBuf::from("a/b/c.txt"), &PathBuf::from("a/b/d.txt"));
        assert_eq!(fixed, PathBuf::from("./d.txt"));
    }

    #[test]
    fn relativize_within_shorter() {
        let fixed = relativize(&PathBuf::from("a/b/c.txt"), &PathBuf::from("a/x/y/z.txt"));
        assert_eq!(fixed, PathBuf::from("../x/y/z.txt"));
    }

    #[test]
    fn relativize_small_path() {
        let fixed = relativize(&PathBuf::from("a/b.txt"), &PathBuf::from("a/x.txt"));
        assert_eq!(fixed, PathBuf::from("./x.txt"));
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct VarRef(usize);
impl VarRef {
    fn pp(&self) -> String {
        format!("var{:}", self.0)
    }
}

// Types are stratified (am I using that term right?) because
// this will enforce we always can name any subrecords/unions.
// The nesting unions/records under any other types must be
// indirect via a reference.
//
// TODO: If further symbolic manipulation is necessary consider
// using the {ann, expr} <-> expr mutual recursion trick to add
// an arbitrary annotation to every node.
#[derive(Debug, Clone, PartialEq)]
pub enum Type0 {
    Var(VarRef),
    Bool,
    Text,
    Natural,
    List(Rc<Type0>),
    Optional(Rc<Type0>),
    StringMap(Rc<Type0>),
    // Relative path, name, unapplied bindings, applied vars
    // invariant: Applied vars monotonically increases
    Reference(PathBuf, String, usize, Vec<VarRef>),
}

fn str_cap(s: &str) -> String {
    format!("{}{}", (&s[..1].to_string()).to_uppercase(), &s[1..])
}

impl Type0 {
    fn indirection(&self, name: String, ctx: &mut Context) -> Type0 {
        let has_name = ctx.has(&name);
        let mut fallthrough = || {
            let name = ctx.fresh(&name);
            ctx.insert(&name, Rc::new(Type::Basic(Rc::new(self.clone()))));
            Type0::Reference(ctx.relative_within(true), ctx.named(&name), 0, Vec::new())
        };
        match self {
            Type0::Reference(_, _, _, _) => {
                // if we already have written this in context reuse it
                if has_name {
                    self.clone()
                } else {
                    fallthrough()
                }
            }
            _ => fallthrough(),
        }
    }

    fn to_variant_name(&self, ctx: &mut Context) -> String {
        match self {
            Type0::Bool => "Boolean".to_string(),
            Type0::Text => "String".to_string(),
            Type0::Natural => "Number".to_string(),
            Type0::List(t_) => format!("List{}", t_.to_variant_name(ctx)),
            Type0::Optional(t_) => format!("Optional{}", t_.to_variant_name(ctx)),
            Type0::StringMap(t_) => format!("StringMap{}", t_.to_variant_name(ctx)),
            Type0::Reference(_, s, _, _) => {
                let p = PathBuf::from(s);
                let mut p_ = p.clone();
                // pop to remove detritus
                while path_to_file(&p_).starts_with("Type")
                    || is_grouping_component(&path_to_file(&p_))
                {
                    p_.pop();
                }
                let prefix = str_cap(&path_to_file(&p_));
                let rest = path_to_file(&p);
                if prefix == "#" {
                    str_cap(&rest)
                } else {
                    format!("{}/{}", prefix, rest)
                }
            }
            Type0::Var(var) => var.pp(),
        }
    }

    fn pp(&self) -> String {
        match self {
            Type0::Bool => "Bool".to_string(),
            Type0::Text => "Text".to_string(),
            Type0::Natural => "Natural".to_string(),
            Type0::List(t_) => format!("List {}", t_.pp()),
            Type0::Optional(t_) => format!("Optional {}", t_.pp()),
            Type0::StringMap(t_) => format!("List {{ mapKey: Text, mapValue: {} }}", t_.pp()),
            Type0::Reference(path, s, _, vars) => {
                let path = PathBuf::from(&format!(
                    "{}/Type",
                    path.to_str().unwrap().to_string().replace("#/", "")
                ));
                let r = relativize(&path, &PathBuf::from(s.replace("#/", "")))
                    .to_str()
                    .unwrap()
                    .to_string();
                /*println!(
                    "In1: {:?}, In2: {:?}, Out: {:?}",
                    path,
                    s.replace("#/", ""),
                    r
                );*/
                format!(
                    "({} {})",
                    r,
                    &vars[..]
                        .iter()
                        .map(|v| v.pp())
                        .collect::<Vec<String>>()
                        .join(" ")
                )
            }
            Type0::Var(var) => var.pp(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Basic(Rc<Type0>),
    // union and record can't recurse directly, they must use references
    // this will enforce we always can name any subrecords/unions
    Union(BTreeMap<String, Option<Rc<Type0>>>),
    Record(BTreeMap<String, Rc<Type0>>),
    // Pi can recurse directly
    // Also, this is sort of a lie, since it's really a value lambda
    Pi(BTreeSet<VarRef>, Rc<Type>),
}
impl Type {
    fn arity(&self) -> usize {
        match self {
            Type::Pi(vs, rest) => vs.len() + rest.arity(),
            _ => 0,
        }
    }

    fn indirection(&self, name: String, ctx: &mut Context) -> Type0 {
        match self {
            Type::Basic(typ) => typ.indirection(name, ctx),
            _ => {
                let name = ctx.fresh(&name);
                ctx.insert(&name, Rc::new(self.clone()));
                Type0::Reference(ctx.relative_within(true), ctx.named(&name), 0, Vec::new())
            }
        }
    }

    fn pp(&self) -> String {
        match self {
            Type::Basic(t0) => t0.pp(),
            Type::Union(map) => {
                let mut chunks = vec!["<".to_string()];
                let mut first = true;
                for (key, val) in map.iter() {
                    if !first {
                        chunks.push("|".to_string());
                    }
                    chunks.push(key.to_string());
                    match val {
                        Some(typ) => {
                            chunks.push(":".to_string());
                            chunks.push(typ.pp());
                        }
                        None => (),
                    };

                    first = false;
                }
                chunks.push(">".to_string());
                chunks.join(" ")
            }
            Type::Record(map) => {
                let mut chunks = vec!["{".to_string()];
                let mut first = true;
                for (key, val) in map.iter() {
                    if !first {
                        chunks.push("\n,".to_string());
                    }
                    chunks.push(key.to_string());
                    chunks.push(":".to_string());
                    chunks.push(val.pp());
                    first = false;
                }
                chunks.push("\n}".to_string());
                chunks.join(" ")
            }
            Type::Pi(vars, t_) => format!("{} {}", pp_pi_prefix(vars), t_.pp()),
        }
    }
}

fn path_to_file(p: &PathBuf) -> String {
    p.file_name().unwrap().to_str().unwrap().to_string()
}

fn is_grouping_component(s: &str) -> bool {
    s == "union" || s == "properties"
}

pub struct Paths {
    path_type: PathBuf,
    path_schema: PathBuf,
    dirpath: PathBuf,
}
impl Paths {
    fn type_to_schema(s: &str) -> String {
        s.replace("/Type", "/Schema")
    }

    fn schema_to_type(s: &str) -> String {
        s.replace("/Schema", "/Type")
    }
}

#[derive(Debug, Clone)]
pub struct Context {
    fresh_: BTreeMap<String, u32>,
    vars: usize,
    path: PathBuf,
    top_level_path: PathBuf,
    data: BTreeMap<String, Rc<Type>>,
}
impl Context {
    fn new() -> Context {
        Context {
            fresh_: BTreeMap::new(),
            vars: 0,
            path: PathBuf::new().join("#"),
            top_level_path: PathBuf::new().join("/"),
            data: BTreeMap::new(),
        }
    }

    fn paths_from_key(key: &str) -> Paths {
        let key = key.replace("#/", "out/");
        let path_type = PathBuf::from(key.clone());
        let path_schema = PathBuf::from(Paths::type_to_schema(&key));
        let path_components = path_to_components(&path_type);
        let (_, dir_components) = path_components.split_last().unwrap();
        let dirpath = PathBuf::from(dir_components.join("/"));
        Paths {
            path_type,
            path_schema,
            dirpath,
        }
    }

    fn new_var(&mut self) -> VarRef {
        let x = self.vars;
        self.vars += 1;
        VarRef(x)
    }

    fn push(&mut self, s: &str) {
        self.path = self.path.join(Path::new(s))
    }

    fn update_top_level_path(&mut self) {
        self.top_level_path = self.path.clone();
    }

    fn relative_within(&self, keep_first_grouping: bool) -> PathBuf {
        // take until last "grouping component"
        // in other words, start from the back and skip until the first one
        let components = path_to_components(&self.path);
        let mut once = true;
        PathBuf::from(
            components
                .iter()
                .rev()
                .skip_while(|r| {
                    if keep_first_grouping {
                        let old = once;
                        once = once && !is_grouping_component(r);
                        old
                    } else {
                        !is_grouping_component(r)
                    }
                })
                .map(|s| s.clone())
                .collect::<Vec<String>>()
                .iter()
                .rev()
                .map(|s| s.clone())
                .collect::<Vec<String>>()
                .join("/"),
        )
    }

    fn pop(&mut self) -> bool {
        self.path.pop()
    }

    fn named(&self, k: &str) -> String {
        let path_ = self.path.join(Path::new(k));
        path_.to_str().unwrap().to_string()
    }

    fn has(&self, k: &str) -> bool {
        let key = self.named(k);
        self.fresh_.get(&key).is_some()
    }

    fn insert(&mut self, k: &str, v: Rc<Type>) {
        let name = self.named(k);
        self.data.insert(name, v);
    }

    fn fresh(&mut self, name: &str) -> String {
        let key = &self.named(name);
        let result = self.fresh_.get(key).map(|v| *v);
        match result {
            Some(x) => {
                self.fresh_.insert(key.to_string(), x + 1);
                format!("Type{}", x)
            }
            None => {
                self.fresh_.insert(key.to_string(), 0);
                "Type".to_string()
            }
        }
    }

    fn file(&self) -> String {
        path_to_file(&self.path)
    }
}

fn unionOrRegular(ctx: &mut Context, ss: &[Schema]) -> Type {
    match ss {
        [Schema::Object(s)] => schemaToTypeTopLevel(ctx, s),
        xs => {
            let mut map: BTreeMap<String, Option<Rc<Type0>>> = BTreeMap::new();
            for x in xs.iter() {
                match x {
                    Schema::Object(s) => {
                        ctx.push("union");
                        let typ = schemaToType(ctx, s);
                        ctx.pop();
                        let _ = map.insert(typ.to_variant_name(ctx), Some(Rc::new(typ)));
                    }
                    _ => panic!("Unimplemented"),
                }
            }
            Type::Union(map)
        }
    }
}

fn objToType(ctx: &mut Context, o: &ObjectValidation) -> Type {
    let mut map: BTreeMap<String, Rc<Type0>> = BTreeMap::new();
    ctx.push("properties");
    let required = o.required.clone();
    for (key, val) in o.properties.iter() {
        let obj = schemaAsObj(val);
        ctx.push(key);
        let name = ctx.file();
        let innerTyp = schemaToType(ctx, obj).indirection(name, ctx);
        ctx.pop();
        // Assume the schema doesn't know what is optional and required (unfortunately)

        map.insert(
            key.clone(),
            Rc::new(if required.contains(key) {
                innerTyp
            } else {
                Type0::Optional(Rc::new(innerTyp))
            }),
        );
    }
    ctx.pop();
    Type::Record(map)
}

fn schemaToTypeTopLevel(ctx: &mut Context, s: &SchemaObject) -> Type {
    let s = s.clone();
    match &s.reference {
        Some(_) => Type::Basic(Rc::new(schemaToType(ctx, &s))),
        None => match &s.instance_type {
            Some(sv) => match sv {
                SingleOrVec::Single(it) => match it {
                    box InstanceType::Object => match s.object {
                        Some(o) => objToType(ctx, &o),
                        None => Type::Basic(Rc::new(schemaToType(ctx, &s))),
                    },
                    _ => Type::Basic(Rc::new(schemaToType(ctx, &s))),
                },
                SingleOrVec::Vec(_) => panic!("Unimplemented"),
            },
            None => {
                // if there is no instance type, we can assume "anyOf"
                match s.subschemas {
                    Some(s) => match s.any_of {
                        None => panic!("Unexpected none for any_of"),
                        Some(schemas) => {
                            let s = schemaAsObj(&schemas[0]);
                            if s.instance_type == Some(SingleOrVec::Single(box InstanceType::Null))
                            {
                                Type::Basic(Rc::new(schemaToType(ctx, s)))
                            } else {
                                unionOrRegular(ctx, schemas.as_slice())
                            }
                        }
                    },
                    None => objToType(ctx, &s.object.expect("We have an object")),
                }
            }
        },
    }
}

fn schemaToType(ctx: &mut Context, s: &SchemaObject) -> Type0 {
    let s = s.clone();
    match s.reference {
        Some(r) => {
            // HACK: commonOptions needs to be stripped from schema for schemars
            // library to properly parse BuildKite's json schema.
            Type0::Reference(
                ctx.path.clone(),
                format!("{}/Type", r.replace("/commonOptions", "")),
                0,
                Vec::new(),
            )
        }
        None => match s.instance_type {
            Some(sv) => match sv {
                SingleOrVec::Single(it) => match it {
                    box InstanceType::Null => panic!("Unimplemented"),
                    box InstanceType::Boolean => Type0::Bool,
                    box InstanceType::Number => Type0::Natural,
                    box InstanceType::Integer => Type0::Integer,
                    box InstanceType::String => Type0::Text,
                    box InstanceType::Array => {
                        let array = s.array.expect("Array has array field");
                        let items = array.items.expect("Array has items field");

                        let innerTyp = match items {
                            SingleOrVec::Single(schema) => schemaToType(ctx, schemaAsObj(&schema)),
                            SingleOrVec::Vec(_) => panic!("Unimplemented"),
                        };
                        Type0::List(Rc::new(innerTyp))
                    }
                    box InstanceType::Object => match s.object {
                        Some(o) => {
                            let name = ctx.file();
                            objToType(ctx, &o).indirection(ctx.fresh(&name), ctx)
                        }
                        None => {
                            let v = ctx.new_var();
                            let name = ctx.file();
                            Type::Pi(
                                {
                                    let mut s = BTreeSet::new();
                                    s.insert(v.clone());
                                    s
                                },
                                Rc::new(Type::Basic(Rc::new(Type0::StringMap(Rc::new(
                                    Type0::Var(v),
                                ))))),
                            )
                            .indirection(ctx.fresh(&name), ctx)
                        }
                    },
                },
                SingleOrVec::Vec(_) => panic!("Unimplemented"),
            },
            None => {
                // if there is no instance type, we can assume "anyOf"
                match s.subschemas {
                    Some(s) => match s.any_of {
                        None => panic!("Unexpected none for any_of"),
                        Some(schemas) => {
                            let s = schemaAsObj(&schemas[0]);
                            if s.instance_type == Some(SingleOrVec::Single(box InstanceType::Null))
                            {
                                let (_, rest) = schemas
                                    .as_slice()
                                    .split_first()
                                    .expect("We already indexed at 0");
                                let inner = unionOrRegular(ctx, rest);
                                let resolved =
                                    inner.indirection(format!("{}NonNull", ctx.file()), ctx);

                                Type0::Optional(Rc::new(resolved))
                            } else {
                                let name = ctx.file();
                                unionOrRegular(ctx, schemas.as_slice())
                                    .indirection(ctx.fresh(&name), ctx)
                            }
                        }
                    },
                    None => {
                        let name = ctx.file();
                        objToType(ctx, &s.object.expect("We have an object"))
                            .indirection(ctx.fresh(&name), ctx)
                    }
                }
            }
        },
    }
}

fn schemaAsObj(s: &Schema) -> &SchemaObject {
    match s {
        Schema::Bool(_) => panic!("Unsupported boolean schema"),
        Schema::Object(s) => s,
    }
}

fn addDefinitionsToCtx(ctx: &mut Context, definitions: &BTreeMap<String, Schema>) {
    ctx.push("definitions");
    for (key, value) in definitions.iter() {
        ctx.push(key);
        ctx.update_top_level_path();
        let innerTyp = schemaToTypeTopLevel(ctx, schemaAsObj(value));
        ctx.pop();

        //match innerTyp {
        //    Type::Record(_) => println!("Skipping record"),
        //    t => {
        ctx.insert(&format!("{}/Type", key), Rc::new(innerTyp));
        //    }
        //};
    }
    ctx.pop();
}

fn rootSchema(ctx: &mut Context, r: &RootSchema) -> Type {
    addDefinitionsToCtx(ctx, &r.definitions);

    let rootRecord = schemaToTypeTopLevel(ctx, &r.schema);

    // println!("Root record {:#?}, ctx {:#?}", rootRecord, ctx);
    rootRecord
}

mod lambda_lift {
    use super::*;

    fn gen_new_vars(ctx: &mut Context, count: usize) -> Vec<VarRef> {
        iter::repeat_with(|| ctx.new_var()).take(count).collect()
    }

    // TODO: Remove this duplication with traits at some point
    pub fn pi_all_expr(vars: &[VarRef], expr: Rc<Expr>) -> Rc<Expr> {
        if vars.len() == 0 {
            expr
        } else {
            let mut new_vars = BTreeSet::new();
            for v in vars {
                new_vars.insert(v.clone());
            }

            match &*expr {
                Expr::Pi(old_vars, rest) => {
                    new_vars.append(&mut old_vars.clone());
                    Rc::new(Expr::Pi(new_vars, rest.clone()))
                }
                _ => Rc::new(Expr::Pi(new_vars, expr.clone())),
            }
        }
    }

    pub fn pi_all(vars: &[VarRef], typ: Rc<Type>) -> Rc<Type> {
        if vars.len() == 0 {
            typ
        } else {
            let mut new_vars = BTreeSet::new();
            for v in vars {
                new_vars.insert(v.clone());
            }

            match &*typ {
                Type::Pi(old_vars, rest) => {
                    new_vars.append(&mut old_vars.clone());
                    Rc::new(Type::Pi(new_vars, rest.clone()))
                }
                _ => Rc::new(Type::Pi(new_vars, typ.clone())),
            }
        }
    }

    // Reload all references and update arities
    // Create local Pi's on the perifery to fully saturate all references
    pub fn lift(ctx: &mut Context, typ: Rc<Type>, new_vars: &mut Vec<VarRef>) -> Rc<Type> {
        fn go(ctx: &mut Context, typ: Rc<Type0>, new_vars: &mut Vec<VarRef>) -> Rc<Type0> {
            use Type0::*;
            match &*typ {
                Bool | Text | Natural | Var(_) => typ,
                List(t_) => Rc::new(List(go(ctx, t_.clone(), new_vars))),
                Optional(t_) => Rc::new(Optional(go(ctx, t_.clone(), new_vars))),
                StringMap(t_) => Rc::new(StringMap(go(ctx, t_.clone(), new_vars))),
                Reference(p0, s, _, vars) => {
                    // safe to unwrap since we've already put all refs in the context
                    let arity = match ctx.data.get(s) {
                        Some(typ) => typ.arity(),
                        None => ctx.data.get(&Paths::schema_to_type(&s)).unwrap().arity(),
                    };
                    // gen the vars
                    let mut vs: Vec<VarRef> = gen_new_vars(ctx, arity - vars.len());
                    {
                        // accumulate them for later
                        let mut vs_ = vs.clone();
                        new_vars.append(&mut vs_);
                    }
                    // apply the new vars to the reference
                    Rc::new(Reference(p0.clone(), s.clone(), arity, {
                        let mut vs_ = vars.clone();
                        vs_.append(&mut vs);
                        vs_
                    }))
                }
            }
        }

        let typ_ = match &*typ {
            Type::Pi(v, typ2) => {
                // recurse and fuse here
                let typ_ = lift(ctx, typ2.clone(), new_vars);
                match &*typ_ {
                    Type::Pi(v2, typ3) => {
                        let mut s = v.clone();
                        let mut v2 = v2.clone();
                        s.append(&mut v2);
                        Rc::new(Type::Pi(s, typ3.clone()))
                    }
                    _ => Rc::new(Type::Pi(v.clone(), typ_.clone())),
                }
            }
            Type::Basic(typ0) => Rc::new(Type::Basic(go(ctx, typ0.clone(), new_vars))),
            Type::Record(map) => {
                let mut map_ = BTreeMap::new();
                for (key, val) in map.iter() {
                    map_.insert(key.clone(), go(ctx, val.clone(), new_vars));
                }
                Rc::new(Type::Record(map_))
            }
            Type::Union(map) => {
                let mut map_: BTreeMap<String, Option<Rc<Type0>>> = BTreeMap::new();
                for (key, val) in map.iter() {
                    map_.insert(key.clone(), val.clone().map(|v| go(ctx, v, new_vars)));
                }
                Rc::new(Type::Union(map_))
            }
        };

        pi_all(new_vars, typ_)
    }

    // Iterate lifting breadth-first across context until a fixed point
    pub fn run(ctx: &mut Context) {
        // TODO: If this is too slow, materialize a "stable" bool instead of doing deep equality
        // each time.
        let mut all_stable = false;
        while !all_stable {
            let mut data_ = ctx.data.clone();
            data_ = data_
                .iter()
                .map(|(k, v)| {
                    let mut vec = Vec::new();
                    (k.clone(), lift(ctx, v.clone(), &mut vec))
                })
                .collect::<BTreeMap<String, Rc<Type>>>();
            all_stable = ctx.data == data_;
            ctx.data = data_;
        }
    }
}

// Simple values for the top-level definitions
pub enum Expr {
    Type(Rc<Type>),
    None_(Rc<Type0>),
    Pi(BTreeSet<VarRef>, Rc<Expr>),
    Record(BTreeMap<String, Rc<Expr>>),
}
impl Expr {
    fn pp(&self) -> String {
        match self {
            Expr::Type(typ) => typ.pp(),
            Expr::Record(map) => {
                let mut chunks = vec!["{".to_string()];
                let mut first = true;
                for (key, val) in map.iter() {
                    if !first {
                        chunks.push("\n,".to_string());
                    }
                    chunks.push(key.to_string());
                    chunks.push("=".to_string());
                    chunks.push(val.pp());
                    first = false;
                }
                chunks.push("\n}".to_string());
                chunks.join(" ")
            }
            Expr::Pi(vars, e) => format!("{} {}", pp_pi_prefix(vars), e.pp()),
            Expr::None_(typ) => format!("None ({})", typ.pp()),
        }
    }
}

fn make_schema(ctx: &mut Context, typ: Rc<Type>) -> Rc<Expr> {
    let fallthrough = Rc::new(Expr::Type(typ.clone()));
    match &*typ {
        Type::Pi(vars, inner_typ) => {
            Rc::new(Expr::Pi(vars.clone(), make_schema(ctx, inner_typ.clone())))
        }
        Type::Record(typ_map) => {
            let mut defaults: BTreeMap<String, Rc<Expr>> = BTreeMap::new();

            for (key, val) in typ_map {
                match &**val {
                    Type0::Optional(optional_typ) => {
                        defaults
                            .insert(key.to_string(), Rc::new(Expr::None_(optional_typ.clone())));
                    }
                    _ => (),
                }
            }

            let mut map: BTreeMap<String, Rc<Expr>> = BTreeMap::new();
            map.insert("Type".to_string(), Rc::new(Expr::Type(typ.clone())));
            map.insert("default".to_string(), Rc::new(Expr::Record(defaults)));
            Rc::new(Expr::Record(map))
        }
        _ => fallthrough,
    }
}

fn generate_top_level(ctx: &mut Context) {
    let mut map: BTreeMap<String, Rc<Expr>> = BTreeMap::new();
    let data_ = ctx.data.clone();
    for (key, _) in data_ {
        if skip_because_blacklist(&key) {
            continue;
        }
        let name = key.replace("#/", "");
        let mut vec = Vec::new();
        let typ = Rc::new(Type::Basic(Rc::new(Type0::Reference(
            PathBuf::from("#/Type"),
            Paths::type_to_schema(&key),
            0,
            Vec::new(),
        ))));
        let typ_ = lambda_lift::lift(ctx, typ.clone(), &mut vec);
        map.insert(name, Rc::new(Expr::Type(lambda_lift::pi_all(&vec, typ_))));
    }
    let expr = Expr::Record(map);

    fs::create_dir_all("out/top_level").unwrap();
    let mut file = File::create("out/top_level/Type").unwrap();
    file.write_all(expr.pp().as_bytes()).unwrap();
}

fn main() {
    let mut ctx = Context::new();

    // load schema
    let file = File::open(PathBuf::new().join("schema.json")).expect("schema");
    let reader = BufReader::new(file);
    let schema: RootSchema = serde_json::from_reader(reader).expect("from_reader");

    let _ = rootSchema(&mut ctx, &schema);

    lambda_lift::run(&mut ctx);
    let data = ctx.data.clone();
    for (key, val) in data {
        let paths = Context::paths_from_key(&key.replace("#/", "out/"));
        let expr = make_schema(&mut ctx, val.clone());

        fs::create_dir_all(paths.dirpath).unwrap();
        let mut file = File::create(paths.path_schema).unwrap();
        file.write_all(expr.pp().as_bytes()).unwrap();
        let mut file2 = File::create(paths.path_type).unwrap();
        file2.write_all(val.pp().as_bytes()).unwrap();
        // println!("wrote {:} with {:}", key, val.pp());
    }

    generate_top_level(&mut ctx)

    // println!("{:#?}", schema);
}
